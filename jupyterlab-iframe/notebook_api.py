from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
import os
import json
import nbformat as nbf
from openai import OpenAI
from uuid import uuid4
from time import sleep

app = Flask(__name__)
CORS(app)  # Allow all origins

NOTEBOOK_PATH = "/home/jovyan/work"
NOTEBOOK_NAME = ""
NOTEBOOK_SAVED = False

# TEXERA DOCUMENTATION

# https://github.com/Texera/texera/wiki/Guide-to-Use-a-Python-UDF
texera_overview = """
You are a robust compiler that takes python code and translates it to our personal workflow enviornment Texera that uses python. 

Texera is a data analytics tool that uses workflows to do machine learning and data analytics computation. User's are able to drag and drop operators and connect their inputs and outputs in a workflow graphical user interface, which the code we are going to create.

    Texera is able to use Python user defined functions. Documentation of a Python UDF in Texera follows:
    Process Data APIs

    There are three APIs to process the data in different units.

        Tuple API.

    class ProcessTupleOperator(UDFOperatorV2):

        def process_tuple(self, tuple_: Tuple, port: int) -> Iterator[Optional[TupleLike]]:
            yield tuple_

    Tuple API takes one input tuple from a port at a time. It returns an iterator of optional TupleLike instances. A TupleLike is any data structure that supports key-value pairs, such as pytexera.Tuple, dict, defaultdict, NamedTuple, etc.

    Tuple API is useful for implementing functional operations which are applied to tuples one by one, such as map, reduce, and filter.

        Table API.

    class ProcessTableOperator(UDFTableOperator):

        def process_table(self, table: Table, port: int) -> Iterator[Optional[TableLike]]:
            yield table

    Table API consumes a Table at a time, which consists of all the tuples from a port. It returns an iterator of optional TableLike instances. A TableLike is a collection of TupleLike, and currently, we support pytexera.Table and pandas.DataFrame as a TableLike instance. More flexible types will be supported down the road.

    Table API is useful for implementing blocking operations that will consume all the data from one port, such as join, sort, and machine learning training.

        Batch API.

    class ProcessBatchOperator(UDFBatchOperator):

        BATCH_SIZE = 10

        def process_batch(self, batch: Batch, port: int) -> Iterator[Optional[BatchLike]]:
            yield batch

    Batch API consumes a batch of tuples at a time. Similar to Table, a Batch is also a collection of Tuples; however, its size is defined by the BATCH_SIZE, and one port can have multiple batches. It returns an iterator of optional BatchLike instances. A BatchLike is a collection of TupleLike, and currently, we support pytexera.Batch and pandas.DataFrame as a BatchLike instance. More flexible types will be supported down the road.

    The Batch API serves as a hybrid API combining the features of both the Tuple and Table APIs. It is particularly valuable for striking a balance between time and space considerations, offering a trade-off that optimizes efficiency.

    All three APIs can return an empty iterator by yield None.

    The template code for a Python UDF follows: MAKE SURE TO USE THE CLASS NAMES AND FUNCTIONS DEFINED, THIS IS A MUST FOR THE PROGRAM TO WORK. SELECT 1 OUT OF THE 3 PROCESSING OPERATOR FUNCTIONS TO BUILD DEPENDINGO ON THE CONTEXT OF CODE TRANSLATION. 
    # Choose from the following templates:
    # 
    # from pytexera import *
    # 
    # class ProcessTupleOperator(UDFOperatorV2):
    #     
    #     @overrides
    #     def process_tuple(self, tuple_: Tuple, port: int) -> Iterator[Optional[TupleLike]]:
    #         yield tuple_
    # 
    # class ProcessBatchOperator(UDFBatchOperator):
    #     BATCH_SIZE = 10 # must be a positive integer
    # 
    #     @overrides
    #     def process_batch(self, batch: Batch, port: int) -> Iterator[Optional[BatchLike]]:
    #         yield batch
    # 
    # class ProcessTableOperator(UDFTableOperator):
    # 
    #     @overrides
    #     def process_table(self, table: Table, port: int) -> Iterator[Optional[TableLike]]:
    #         yield table
"""

# https://github.com/Texera/texera/blob/1fa249a9d55d4dcad36d93e093c2faed5c4434f0/core/amber/src/main/python/core/models/tuple.py
tuple_documentation = """
Here is tuple.py documentation to help generate a python udf if needed: 

import ctypes
import struct
import typing
from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, List, Iterator, Callable

from typing_extensions import Protocol, runtime_checkable
import pandas
import pickle
import pyarrow
from loguru import logger
from pandas._libs.missing import checknull

from .schema.attribute_type import TO_PYOBJECT_MAPPING, AttributeType
from .schema.field import Field
from .schema.schema import Schema


@runtime_checkable
class TupleLike(Protocol):
    def __getitem__(self, item: typing.Union[str, int]) -> Field: ...

    def __setitem__(self, key: typing.Union[str, int], value: Field) -> None: ...


@dataclass
class InputExhausted:
    pass


class ArrowTableTupleProvider:
    # This class provides "view"s for tuple from a pyarrow.Table.

    def __init__(self, table: pyarrow.Table):
        # Construct a provider from a pyarrow.Table.
        # Keep the current chunk and tuple idx as its state.
        self._table = table
        self._current_idx = 0
        self._current_chunk = 0

    def __iter__(self) -> Iterator[Callable]:
        # Return itself as it is iterable.
        return self

    def __next__(self) -> Callable:
        # Provide the field accessor of the next tuple.
        # If current chunk is exhausted, move to the first tuple of the next chunk.
        if self._current_idx >= len(self._table.column(0).chunks[self._current_chunk]):
            self._current_idx = 0
            self._current_chunk += 1
            if self._current_chunk >= self._table.column(0).num_chunks:
                raise StopIteration

        chunk_idx = self._current_chunk
        tuple_idx = self._current_idx

        def field_accessor(field_name: str) -> Field:
            # Retrieve the field value by a given field name.
            # This abstracts and hides the underlying implementation of the tuple data
            # storage from the user.
            value = self._table.column(field_name).chunks[chunk_idx][tuple_idx].as_py()
            field_type = self._table.schema.field(field_name).type

            # for binary types, convert pickled objects back.
            if (
                field_type == pyarrow.binary()
                and value is not None
                and value[:6] == b"pickle"
            ):
                value = pickle.loads(value[10:])
            return value

        self._current_idx += 1
        return field_accessor


def double_to_long(value: float) -> int:
    # Convert a double value into a long value.
    # :param value: A double (Python float) value.
    # :return: The converted long (Python int) value.
    # Pack the double value into a binary string of 8 bytes
    packed_value = struct.pack("d", value)
    # Unpack the binary string to a 64-bit integer (int in Python 3)
    long_value = struct.unpack("Q", packed_value)[0]
    return long_value


def int_32(value: int) -> int:
    # Convert a Python int (unbounded) to a 32-bit int with overflow.
    # :param value: A Python int value.
    # :return: The converted 32-bit integer, with overflow.
    return ctypes.c_int32(value).value


def java_hash_bool(value: bool) -> int:
    # Java's hash function for a boolean value.
    # :param value: A boolean value.
    # :return: Java's hash value in a 32-bit integer.
    return 1231 if value else 1237


def java_hash_long(value: int) -> int:
    # Java's hash function for a long value.
    # :param value: A long (Python int) value.
    # :return: Java's hash value in a 32-bit integer.
    return int_32(value ^ (value >> 32))


def java_hash_bytes(bytes: Iterator[int], init: int, salt: int):
    # Java's hash function for an array of bytes.
    # :param bytes: An iterator of int (byte) values.
    # :param init: An init hash value.
    # :param salt: A hash salt value.
    # :return: Java's hash value in a 32-bit integer.
    h = init
    for b in bytes:
        h = int_32(salt * h + b)
    return h


class Tuple:
    # Lazy-Tuple implementation.

    def __init__(
        self,
        tuple_like: typing.Optional["TupleLike"] = None,
        schema: typing.Optional[Schema] = None,
    ):
        # Construct a lazy-tuple with given TupleLike object. If the field value is a
        # accessor callable, the actual value is fetched upon first reference.

        # :param tuple_like: in which the field value could be the actual value in
        #     memory, or a callable accessor.
        assert len(tuple_like) != 0
        self._field_data: "OrderedDict[str, Field]"
        if isinstance(tuple_like, Tuple):
            self._field_data = tuple_like._field_data
        elif isinstance(tuple_like, pandas.Series):
            self._field_data = OrderedDict(tuple_like.to_dict())
        else:
            self._field_data = OrderedDict(tuple_like) if tuple_like else OrderedDict()
        self._schema: typing.Optional[Schema] = schema
        if self._schema:
            self.finalize(schema)

    def __getitem__(self, item: typing.Union[int, str]) -> Field:
        # Get a field value with given item. If the value is an accessor, fetch it from
        # the accessor.

        # :param item: field name or field index
        # :return: field value
        assert isinstance(
            item, (int, str)
        ), "field can only be retrieved by index or name"

        if isinstance(item, int):
            item: str = self.get_field_names()[item]

        if (
            callable(self._field_data[item])
            and getattr(self._field_data[item], "__name__", "Unknown")
            == "field_accessor"
        ):
            # evaluate the field now
            field_accessor = self._field_data[item]
            self._field_data[item] = field_accessor(field_name=item)
        return self._field_data[item]

    def __setitem__(self, field_name: str, field_value: Field) -> None:
        # Set a field with the given value.
        # :param field_name
        # :param field_value
        assert isinstance(field_name, str), "field can only be set by name"
        assert not callable(field_value), "field cannot be of type callable"
        self._field_data[field_name] = field_value

    def as_series(self) -> pandas.Series:
        # Convert the tuple to Pandas series format
        return pandas.Series(self.as_dict())

    def as_dict(self) -> "OrderedDict[str, Field]":
        # Return a dictionary copy of this tuple.
        # Fields will be fetched from accessor if absent.
        # :return: dict with all the fields
        # evaluate all the fields now
        for i in self.get_field_names():
            self.__getitem__(i)
        return deepcopy(self._field_data)

    def as_key_value_pairs(self) -> List[typing.Tuple[str, Field]]:
        return [(k, v) for k, v in self.as_dict().items()]

    def get_field_names(self) -> typing.Tuple[str]:
        return tuple(map(str, self._field_data.keys()))

    def get_fields(self, output_field_names=None) -> typing.Tuple[Field, ...]:
        # Get values from tuple for selected fields.
        if output_field_names is None:
            output_field_names = self.get_field_names()
        return tuple(self[i] for i in output_field_names)

    def finalize(self, schema: Schema) -> None:
        # Finalizes a Tuple by adding a schema to it. This convert all Fields into the
        # AttributeType defined in the Schema and make the Tuple immutable.

        # A Tuple can have no Schema initially. The types of Fields are not restricted.
        # This is to provide the maximum flexibility for users to construct Tuples as
        # they wish. When a Schema is added, the Tuple is finalized to match the Schema.

        # :param schema: target Schema to finalize the Tuple.
        # :return:
        self.cast_to_schema(schema)
        self.validate_schema(schema)
        self._schema = schema

    def cast_to_schema(self, schema: Schema) -> None:
        # Safely cast each field value to match the target schema.
        # If failed, the value will stay not changed.

        # This current conducts two kinds of casts:
        #     1. cast NaN to None;
        #     2. cast any object to bytes (using pickle).
        # :param schema: The target Schema that describes the target AttributeType to
        #     cast.
        # :return:
        for field_name in self.get_field_names():
            try:
                field_value: Field = self[field_name]

                # convert NaN to None to support null value conversion
                if checknull(field_value):
                    self[field_name] = None

                if field_value is not None:
                    field_type = schema.get_attr_type(field_name)
                    if field_type == AttributeType.BINARY and not isinstance(
                        field_value, bytes
                    ):
                        self[field_name] = b"pickle    " + pickle.dumps(field_value)
            except Exception as err:
                # Surpass exceptions during cast.
                # Keep the value as it is if the cast fails, and continue to attempt
                # on the next one.
                logger.warning(err)
                continue

    def validate_schema(self, schema: Schema) -> None:
        # Checks if the field values in the Tuple matches the expected Schema.
        # :param schema: Schema
        # :return:

        schema_fields = schema.get_attr_names()
        tuple_fields = self.get_field_names()
        expected_but_missing = set(schema_fields) - set(tuple_fields)
        unexpected = set(tuple_fields) - set(schema_fields)
        if expected_but_missing:
            raise KeyError(
                f"field{'' if len(expected_but_missing) == 1 else 's'} "
                f"{', '.join(map(repr, expected_but_missing))} "
                f"{'is' if len(expected_but_missing) == 1 else 'are'} "
                f"expected but missing in the {self}."
            )

        if unexpected:
            raise KeyError(
                f"{self} contains {'an' if len(unexpected) == 1 else ''} unexpected "
                f"field{'' if len(unexpected) == 1 else 's'}: "
                f"{', '.join(map(repr, unexpected))}."
            )

        for field_name, field_value in self.as_key_value_pairs():
            expected = schema.get_attr_type(field_name)
            if not isinstance(
                field_value, (TO_PYOBJECT_MAPPING.get(expected), type(None))
            ):
                raise TypeError(
                    f"Unmatched type for field '{field_name}', expected {expected}, "
                    f"got {field_value} ({type(field_value)}) instead."
                )

    def get_partial_tuple(self, attribute_names: List[str]) -> "Tuple":
        # Creates a partial Tuple with fields specified by the attribute names.

        # :param attribute_names: A list of attribute names for which to create the
        #                         partial tuple.
        # :return: A new Tuple instance containing only the specified fields,
        #         preserving the order specified by the attribute names.
        assert self._schema is not None
        schema = self._schema.get_partial_schema(attribute_names)
        new_raw_tuple = OrderedDict()
        for name in attribute_names:
            new_raw_tuple[name] = self[name]
        return Tuple(new_raw_tuple, schema=schema)

    def __iter__(self) -> Iterator[Field]:
        return iter(self.get_fields())

    def __str__(self) -> str:
        content = ", ".join(
            [repr(key) + ": " + repr(value) for key, value in self.as_key_value_pairs()]
        )
        return f"Tuple[{content}]"

    __repr__ = __str__

    def __eq__(self, other: Any) -> bool:
        return (
            isinstance(other, Tuple)
            and self.get_field_names() == other.get_field_names()
            and all(self[i] == other[i] for i in self.get_field_names())
        )

    def __ne__(self, other) -> bool:
        return not self.__eq__(other)

    def __len__(self) -> int:
        return len(self._field_data)

    def __contains__(self, __x: object) -> bool:
        return __x in self._field_data

    def __hash__(self) -> int:
        # Aligned with Java's built-in hash algorithm implementation described in
        # _Josh Bloch's Effective Java_.
        # This algorithm is taken by
        #  - Built-in Java (java.util.Objects.hash)
        #  - Guava (com.google.common.base.Objects.hashCode)
        # :return: A 32-bit integer value.
        result = 1
        salt = 31  # for ease of optimization

        mapping = {
            AttributeType.BOOL: lambda f: java_hash_bool(f),
            AttributeType.INT: lambda f: int_32(f),
            AttributeType.LONG: lambda f: java_hash_long(f),
            AttributeType.DOUBLE: lambda f: java_hash_long(double_to_long(f)),
            AttributeType.STRING: lambda f: java_hash_bytes(map(ord, f), 0, salt),
            AttributeType.TIMESTAMP: lambda f: java_hash_long(int(f.timestamp())),
            AttributeType.BINARY: lambda f: java_hash_bytes(f, 1, salt),
        }

        for name, field in self.as_key_value_pairs():
            attr_type = self._schema.get_attr_type(name)
            if field is None:
                hash_value = 0
            else:
                hash_value = mapping[attr_type](field)
            result = result * salt + hash_value

        return int_32(result)

"""

# https://github.com/Texera/texera/blob/1fa249a9d55d4dcad36d93e093c2faed5c4434f0/core/amber/src/main/python/core/models/table.py
table_documentation = """
Here is table.py documentation to help generate a python udf if needed: 

from typing import Iterator, TypeVar, List

from pampy import match
import pandas

from core.models import Tuple, TupleLike

TableLike = TypeVar("TableLike", pandas.DataFrame, List[TupleLike])

class Table(pandas.DataFrame):
    @staticmethod
    def from_table(table):
        return table

    @staticmethod
    def from_data_frame(df):
        return df

    @staticmethod
    def from_tuple_likes(tuple_likes: Iterator[TupleLike]):
        # TODO: currently only validate all Tuples have the same fields.
        #  should validate types as well
        column_names = None
        records = []
        for tuple_like in tuple_likes:
            tuple_ = Tuple(tuple_like)
            field_names = tuple_.get_field_names()

            if column_names is not None:
                assert field_names == column_names
            else:
                column_names = field_names

            records.append(tuple_.get_fields())

        return pandas.DataFrame.from_records(records, columns=column_names)

    def __init__(self, table_like):
        df: pandas.DataFrame

        if isinstance(table_like, Table):
            df = self.from_table(table_like)
        elif isinstance(table_like, pandas.DataFrame):
            df = self.from_data_frame(table_like)
        elif isinstance(table_like, list):
            # only supports List[TupleLike]
            df = self.from_tuple_likes(table_like)
        else:
            raise TypeError(f"unsupported tablelike type {type(table_like)}")
        super().__init__(df)

    def as_tuples(self) -> Iterator[Tuple]:
        # Convert rows of the table into Tuples, and returning an iterator of Tuples
        # following their row index order.
        # :return:
        for raw_tuple in self.itertuples(index=False, name=None):
            yield Tuple(dict(zip(self.columns, raw_tuple)))

    def __eq__(self, other: "Table") -> bool:
        if isinstance(other, Table):
            return all(a == b for a, b in zip(self.as_tuples(), other.as_tuples()))
        else:
            return super().__eq__(other).all()

def all_output_to_tuple(output) -> Iterator[Tuple]:
    # Convert all kinds of types into Tuples.
    # :param output:
    # :return:
    yield from match(
        output,
        None,
        iter([None]),
        Table,
        lambda x: x.as_tuples(),
        pandas.DataFrame,
        lambda x: Table(x).as_tuples(),
        List[TupleLike],
        lambda x: (Tuple(t) for t in x),
        TupleLike,
        lambda x: iter([Tuple(x)]),
        Tuple,
        lambda x: iter([x]),
    )
"""

# https://github.com/Texera/texera/blob/42d803310c180978a9f02992f0e05556796b293c/core/amber/src/main/python/core/models/operator.py
operator_documentation = """
Here is operator.py documentation to help generate a python udf if needed: 

from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Iterator, List, Mapping, Optional, Union, MutableMapping

import overrides
import pandas

from deprecated import deprecated

from . import InputExhausted, Table, TableLike, Tuple, TupleLike, Batch, BatchLike
from .table import all_output_to_tuple

class Operator(ABC):
    # Abstract base class for all operators.

    __internal_is_source: bool = False

    @property
    @overrides.final
    def is_source(self) -> bool:
        # Whether the operator is a source operator. Source operators generate output
        # Tuples without having input Tuples.

        # :return:
        return self.__internal_is_source

    @is_source.setter
    @overrides.final
    def is_source(self, value: bool) -> None:
        self.__internal_is_source = value

    def open(self) -> None:
        # Open a context of the operator. Usually can be used for loading/initiating some
        # resources, such as a file, a model, or an API client.
        pass

    def close(self) -> None:
        # Close the context of the operator.
        pass

class TupleOperatorV2(Operator):
    # Base class for tuple-oriented operators. A concrete implementation must
    # be provided upon using.

    @abstractmethod
    def process_tuple(self, tuple_: Tuple, port: int) -> Iterator[Optional[TupleLike]]:
        # Process an input Tuple from the given link.

        # :param tuple_: Tuple, a Tuple from an input port to be processed.
        # :param port: int, input port index of the current Tuple.
        # :return: Iterator[Optional[TupleLike]], producing one TupleLike object at a
        #     time, or None.
        yield

    def on_finish(self, port: int) -> Iterator[Optional[TupleLike]]:
        # Callback when one input port is exhausted.

        # :param port: int, input port index of the current exhausted port.
        # :return: Iterator[Optional[TupleLike]], producing one TupleLike object at a
        #     time, or None.
        yield

class SourceOperator(TupleOperatorV2):
    __internal_is_source = True

    @abstractmethod
    def produce(self) -> Iterator[Union[TupleLike, TableLike, None]]:
        # Produce Tuples or Tables. Used by the source operator only.

        # :return: Iterator[Union[TupleLike, TableLike, None]], producing
        #     one TupleLike object, one TableLike object, or None, at a time.
        yield

    @overrides.final
    def on_finish(self, port: int) -> Iterator[Optional[TupleLike]]:
        # TODO: change on_finish to output Iterator[Union[TupleLike, TableLike, None]]
        for i in self.produce():
            yield from all_output_to_tuple(i)

    @overrides.final
    def process_tuple(self, tuple_: Tuple, port: int) -> Iterator[Optional[TupleLike]]:
        yield

class BatchOperator(TupleOperatorV2):
    # Base class for batch-oriented operators. A concrete implementation must
    # be provided upon using.

    BATCH_SIZE: int = 10  # must be a positive integer

    def __init__(self):
        super().__init__()
        self.__batch_data: MutableMapping[int, List[Tuple]] = defaultdict(list)
        self._validate_batch_size(self.BATCH_SIZE)

    @staticmethod
    @overrides.final
    def _validate_batch_size(value):
        if value is None:
            raise ValueError("BATCH_SIZE cannot be None.")
        if type(value) is not int:
            raise ValueError("BATCH_SIZE cannot be {type(value))}.")
        if value <= 0:
            raise ValueError("BATCH_SIZE should be positive.")

    @overrides.final
    def process_tuple(self, tuple_: Tuple, port: int) -> Iterator[Optional[TupleLike]]:
        self.__batch_data[port].append(tuple_)
        if (
            self.BATCH_SIZE is not None
            and len(self.__batch_data[port]) >= self.BATCH_SIZE
        ):
            yield from self._process_batch(port)

    @overrides.final
    def _process_batch(self, port: int) -> Iterator[Optional[BatchLike]]:
        batch = Batch(
            pandas.DataFrame(
                [
                    self.__batch_data[port].pop(0).as_series()
                    for _ in range(min(len(self.__batch_data[port]), self.BATCH_SIZE))
                ]
            )
        )
        for output_batch in self.process_batch(batch, port):
            if output_batch is not None:
                if isinstance(output_batch, pandas.DataFrame):
                    # TODO: integrate into Batch as a helper function.
                    # convert from Batch to Tuple, only supports pandas.DataFrames for
                    # now.
                    for _, output_tuple in output_batch.iterrows():
                        yield output_tuple
                else:
                    yield output_batch

    @overrides.final
    def on_finish(self, port: int) -> Iterator[Optional[BatchLike]]:
        while len(self.__batch_data[port]) != 0:
            yield from self._process_batch(port)

    @abstractmethod
    def process_batch(self, batch: Batch, port: int) -> Iterator[Optional[BatchLike]]:
        # Process an input Batch from the given link. The Batch is represented as a
        # pandas.DataFrame.

        # :param batch: Batch, a batch to be processed.
        # :param port: int, input port index of the current Batch.
        # :return: Iterator[Optional[BatchLike]], producing one BatchLike object at a
        #     time, or None.
        yield

class TableOperator(TupleOperatorV2):
    # Base class for table-oriented operators. A concrete implementation must
    # be provided upon using.

    def __init__(self):
        super().__init__()
        self.__internal_is_source: bool = False
        self.__table_data: Mapping[int, List[Tuple]] = defaultdict(list)

    @overrides.final
    def process_tuple(self, tuple_: Tuple, port: int) -> Iterator[Optional[TupleLike]]:
        self.__table_data[port].append(tuple_)
        yield

    def on_finish(self, port: int) -> Iterator[Optional[TableLike]]:
        table = Table(self.__table_data[port])
        yield from self.process_table(table, port)

    @abstractmethod
    def process_table(self, table: Table, port: int) -> Iterator[Optional[TableLike]]:
        # Process an input Table from the given link. The Table is represented as a
        # pandas.DataFrame.

        # :param table: Table, a table to be processed.
        # :param port: int, input port index of the current Tuple.
        # :return: Iterator[Optional[TableLike]], producing one TableLike object at a
        #     time, or None.
        yield

@deprecated(reason="Use TupleOperatorV2 instead")
class TupleOperator(Operator):
    # Base class for tuple-oriented operators. A concrete implementation must
    # be provided upon using.

    @abstractmethod
    def process_tuple(
        self, tuple_: Union[Tuple, InputExhausted], input_: int
    ) -> Iterator[Optional[TupleLike]]:
        # Process an input Tuple from the given link.

        # :param tuple_: Union[Tuple, InputExhausted], either
        #                 1. a Tuple from a link to be processed;
        #                 2. an InputExhausted indicating no more data from this link.
        # :param input_: int, input index of the current Tuple.
        # :return: Iterator[Optional[TupleLike]], producing one TupleLike object at a
        #     time, or None.
        yield

    def on_finish(self, port: int) -> Iterator[Optional[TupleLike]]:
        # For backward compatibility.
        yield from self.process_tuple(InputExhausted(), input_=port)

"""

udf_input_port_documentation = """
Python UDF operators support multiple input and output ports, allowing a single operator to receive different types of data from various upstream operators. In the process_tuple(self, tuple_: Tuple, port: int) function in ProcessTupleOperator and the process_table(self, table: Table, port: int) function in ProcessTableOperator, the port parameter indicates the input port. The port numbers are assigned in order, starting from 0 to N, from top to bottom. When input data have different schemas, it is necessary to assign them to different input ports. However, if all input data share the same schema, additional ports are not required. In both ProcessTupleOperator and ProcessTableOperator, there is an on_finish(self, port: int) function that is executed only after all the tuples from the specified port are processed.

Using this knowledge, for situations where multiple upstream UDFs act as input to a single UDF, we can introduce an intermediary UDF that collects all of the input data and reformats it into a single table, which is then passed as input to the original next downstream UDF. When it is necessary for this to occur in your translation from notebook to UDFs, include the intermediary UDF and make sure that it and the next operator that uses its output is formatted correctly and handles the data transfer properly.
"""

example_of_good_conversion = """
Here is an example of python code translated into a compatible Texera UDF that gives output that abides the output schema compatible with the Texera workflow operators for tuples. Other operators do not always follow this strict format, but the yielding output structure is important.

Python Code (high level idea): We have a python code that given some data, we limit the number of data.

Texera Operator code: 
from pytexera import *

class ProcessTupleOperator(UDFOperatorV2):
    def __init__(self):
        self.limit = 10
        self.count = 0
    @overrides
    def process_tuple(self, tuple_: Tuple, port: int) -> Iterator[Optional[TupleLike]]:
        if(self.count < self.limit):
            self.count += 1
            yield tuple_

"""

visualizer_documentation = """
Texera requires a unique way of visualizing certain visualizations from ML libraries. Here is an example to follow for building visualization code. Note: DO NOT CHANGE OPERATOR CLASS NAME NOR ITS FUNCTIONS!!! Use the class and function names as shown in ProcessTupleOperator, ProcessTableOperator, and ProcessBatchOperator. Do not change the class names, function names, or input parameters. Use the ones that make sense and split the code meaningfully as instructed. Each visualization should be divided into separate Python UDFs, as there is only one yield per UDF code operator.

from pytexera import *
import plotly.express as px
import plotly.io

class ProcessTableOperator(UDFTableOperator):
    def render_error(self, error_msg):
        return '''<h1>Histogram chart is not available.</h1>
                  <p>Reason is: {} </p>
               '''.format(error_msg)

    @overrides
    def process_table(self, table: Table, port: int) -> Iterator[Optional[TableLike]]:
        if table.empty:
           yield {'html-content': self.render_error('input table is empty.')}
           return

        # Creating the Plotly figure based on provided column names and options
        fig = px.histogram(table, x='value', text_auto=True, color='color', facet_col='separateBy', marginal='marginal')
        fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))

        # Convert fig to HTML content
        html = plotly.io.to_html(fig, include_plotlyjs='cdn', auto_play=False)
        yield {'html-content': html}
"""

example_of_multiple_udf_conversion = """
Here is an example of breaking up python code into multiple Texera UDFs. Format your response structure exactly like the given example. The "code" key contains a dictionary of the UDF ID's with their respective code. The "edges" key contains a list of pairs that contains the connections between UDFs. The "outputs" key contains a dictionary of the UDF ID's with a list of variable names that they yield in the UDF code. The UDFs can branch and merge, it does not have to be a linear chain depending on your implementation.

Original Code:
```python
# START CELL1
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
# END CELL1

# START CELL2
# Load the dataset
file_path = 'diabetes.csv'
data = pd.read_csv(file_path)
# END CELL2

# START CELL3
# Remove duplicate rows
data = data.drop_duplicates()

# Remove rows with null values
data = data.dropna()
# END CELL3

# START CELL4
# Print the minimum, maximum, and mean for all fields
print("Minimum values:\n", data.min())
print("\nMaximum values:\n", data.max())
print("\nMean values:\n", data.mean())
# END CELL 4

# START CELL5
# Create a boxplot for the 'Pregnancies' field
plt.figure(figsize=(8, 6))
plt.boxplot(data['Pregnancies'], vert=False, patch_artist=True)
plt.title('Boxplot of Pregnancies')
plt.xlabel('Number of Pregnancies')
plt.show()
# END CELL5

# START CELL6
# Separate features and target variable
X = data.drop('Outcome', axis=1)
y = data['Outcome']
# END CELL6

# START CELL7
# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# END CELL7

# START CELL8
# Train Random Forest model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_pred)
print(f"Random Forest Accuracy: {rf_accuracy:.2%}")
# END CELL8

# START CELL9
# Train SVM model
svm_model = SVC(random_state=42)
svm_model.fit(X_train, y_train)
svm_pred = svm_model.predict(X_test)
svm_accuracy = accuracy_score(y_test, svm_pred)
print(f"SVM Accuracy: {svm_accuracy:.2%}")
# END CELL9

# START CELL10
# Train Decision Tree model
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)
dt_accuracy = accuracy_score(y_test, dt_pred)
print(f"Decision Tree Accuracy: {dt_accuracy:.2%}")
# END CELL10

# START CELL11
# Train Logistic Regression model
lr_model = LogisticRegression(random_state=42)
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)
lr_accuracy = accuracy_score(y_test, lr_pred)
print(f"Logistic Regression Accuracy: {lr_accuracy:.2%}")
# END CELL11
```

Texera UDF conversion:
```json
{
    "code": {
        "UDF1": "# UDF1\nfrom pytexera import *\nimport pandas as pd\nfrom typing import Iterator, Optional\n\nclass ProcessTableOperator(UDFTableOperator):\n\n    @overrides\n    def process_table(self, table: Table, port: int) -> Iterator[Optional[TableLike]]:\n        # Remove duplicate rows\n        data = table.drop_duplicates()\n\n        # Remove rows with null values\n        data = data.dropna()\n\n        # Calculate statistics\n        min_values = data.min()\n        max_values = data.max()\n        mean_values = data.mean()\n\n        # Create a DataFrame to yield\n        result_table = pd.DataFrame({\n            'min_values': [min_values],\n            'max_values': [max_values],\n            'mean_values': [mean_values],\n            'data': [data]\n        })\n\n        yield Table(result_table)",
        "UDF2": "# UDF2\nfrom pytexera import *\nimport pandas as pd\nimport plotly.express as px\nimport plotly.io\nfrom typing import Iterator, Optional\n\nclass ProcessTableOperator(UDFTableOperator):\n    def render_error(self, error_msg):\n        return '''<h1>Boxplot is not available.</h1>\n                  <p>Reason is: {} </p>\n               '''.format(error_msg)\n\n    @overrides\n    def process_table(self, table: Table, port: int) -> Iterator[Optional[TableLike]]:\n        data = table['data'].iloc[0]\n\n        if data.empty:\n            yield {'html-content': self.render_error('input table is empty.')}\n            return\n\n        # Create a boxplot for the 'Pregnancies' field\n        fig = px.box(data, x='Pregnancies')\n        fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))\n\n        # Convert fig to HTML content\n        html = plotly.io.to_html(fig, include_plotlyjs='cdn', auto_play=False)\n        yield {'html-content': html}",
        "UDF3": "# UDF3\nfrom pytexera import *\nimport pandas as pd\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.preprocessing import StandardScaler\nfrom typing import Iterator, Optional\n\nclass ProcessTableOperator(UDFTableOperator):\n\n    @overrides\n    def process_table(self, table: Table, port: int) -> Iterator[Optional[TableLike]]:\n        data = table['data'].iloc[0]\n\n        # Separate features and target variable\n        X = data.drop('Outcome', axis=1)\n        y = data['Outcome']\n\n        # Split data into training and testing sets (80% train, 20% test)\n        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n\n        scaler = StandardScaler()\n        X_train = scaler.fit_transform(X_train)\n        X_test = scaler.transform(X_test)\n\n        # Create a DataFrame to yield\n        result_table = pd.DataFrame({\n            'X_train': [X_train], 'X_test': [X_test],\n            'y_train': [y_train], 'y_test': [y_test]\n        })\n\n        yield Table(result_table)",
        "UDF4": "# UDF4\nfrom pytexera import *\nimport pandas as pd\nfrom sklearn.ensemble import RandomForestClassifier\nfrom sklearn.metrics import accuracy_score\nfrom typing import Iterator, Optional\n\nclass ProcessTableOperator(UDFTableOperator):\n\n    @overrides\n    def process_table(self, table: Table, port: int) -> Iterator[Optional[TableLike]]:\n        X_train = table['X_train'].iloc[0]\n        y_train = table['y_train'].iloc[0]\n        X_test = table['X_test'].iloc[0]\n        y_test = table['y_test'].iloc[0]\n\n        # Train Random Forest model\n        rf_model = RandomForestClassifier(random_state=42)\n        rf_model.fit(X_train, y_train)\n        rf_pred = rf_model.predict(X_test)\n        rf_accuracy = accuracy_score(y_test, rf_pred)\n\n        # Create a DataFrame to yield\n        result_table = pd.DataFrame({\n            'rf_model': [rf_model],\n            'rf_accuracy': [rf_accuracy],\n            'X_test': [X_test],\n            'y_test': [y_test]\n        })\n\n        yield Table(result_table)",
        "UDF5": "# UDF5\nfrom pytexera import *\nimport pandas as pd\nfrom sklearn.svm import SVC\nfrom sklearn.metrics import accuracy_score\nfrom typing import Iterator, Optional\n\nclass ProcessTableOperator(UDFTableOperator):\n\n    @overrides\n    def process_table(self, table: Table, port: int) -> Iterator[Optional[TableLike]]:\n        X_train = table['X_train'].iloc[0]\n        y_train = table['y_train'].iloc[0]\n        X_test = table['X_test'].iloc[0]\n        y_test = table['y_test'].iloc[0]\n\n        # Train SVM model\n        svm_model = SVC(random_state=42)\n        svm_model.fit(X_train, y_train)\n        svm_pred = svm_model.predict(X_test)\n        svm_accuracy = accuracy_score(y_test, svm_pred)\n\n        # Create a DataFrame to yield\n        result_table = pd.DataFrame({\n            'svm_model': [svm_model],\n            'svm_accuracy': [svm_accuracy],\n            'X_test': [X_test],\n            'y_test': [y_test]\n        })\n\n        yield Table(result_table)",
        "UDF6": "# UDF6\nfrom pytexera import *\nimport pandas as pd\nfrom sklearn.tree import DecisionTreeClassifier\nfrom sklearn.metrics import accuracy_score\nfrom typing import Iterator, Optional\n\nclass ProcessTableOperator(UDFTableOperator):\n\n    @overrides\n    def process_table(self, table: Table, port: int) -> Iterator[Optional[TableLike]]:\n        X_train = table['X_train'].iloc[0]\n        y_train = table['y_train'].iloc[0]\n        X_test = table['X_test'].iloc[0]\n        y_test = table['y_test'].iloc[0]\n\n        # Train Decision Tree model\n        dt_model = DecisionTreeClassifier(random_state=42)\n        dt_model.fit(X_train, y_train)\n        dt_pred = dt_model.predict(X_test)\n        dt_accuracy = accuracy_score(y_test, dt_pred)\n\n        # Create a DataFrame to yield\n        result_table = pd.DataFrame({\n            'dt_model': [dt_model],\n            'dt_accuracy': [dt_accuracy],\n            'X_test': [X_test],\n            'y_test': [y_test]\n        })\n\n        yield Table(result_table)",
        "UDF7": "# UDF7\nfrom pytexera import *\nimport pandas as pd\nfrom sklearn.linear_model import LogisticRegression\nfrom sklearn.metrics import accuracy_score\nfrom typing: Iterator, Optional\n\nclass ProcessTableOperator(UDFTableOperator):\n\n    @overrides\n    def process_table(self, table: Table, port: int) -> Iterator[Optional[TableLike]]:\n        X_train = table['X_train'].iloc[0]\n        y_train = table['y_train'].iloc[0]\n        X_test = table['X_test'].iloc[0]\n        y_test = table['y_test'].iloc[0]\n\n        # Train Logistic Regression model\n        lr_model = LogisticRegression(random_state=42)\n        lr_model.fit(X_train, y_train)\n        lr_pred = lr_model.predict(X_test)\n        lr_accuracy = accuracy_score(y_test, lr_pred)\n\n        # Create a DataFrame to yield\n        result_table = pd.DataFrame({\n            'lr_model': [lr_model],\n            'lr_accuracy': [lr_accuracy],\n            'X_test': [X_test],\n            'y_test': [y_test]\n        })\n\n        yield Table(result_table)"
    },
    "edges": [
        ["UDF1", "UDF2"],
        ["UDF1", "UDF3"],
        ["UDF3", "UDF4"],
        ["UDF3", "UDF5"],
        ["UDF3", "UDF6"],
        ["UDF3", "UDF7"]
    ],
    "outputs": {
        "UDF1": ["min_values", "max_values", "mean_values", "data"],
        "UDF2": ["html-content"],
        "UDF3": ["X_train", "X_test", "y_train", "y_test"],
        "UDF4": ["rf_model", "rf_accuracy", "X_test", "y_test"],
        "UDF5": ["svm_model", "svm_accuracy", "X_test", "y_test"],
        "UDF6": ["dt_model", "dt_accuracy", "X_test", "y_test"],
        "UDF7": ["lr_model", "lr_accuracy", "X_test", "y_test"]
    }
}
```
"""

example_of_mapping = """
Here is an example of a mapping generated between the given example Python code and the Texera UDFs using their CELL and UDF IDs. Cell IDs are designated by the UUID following '# START'. The format should be kept the same.
{
    "UDF1": [
        "CEll3",
        "CELL4"
    ],
    "UDF2": [
        "CELL5"
    ],
    "UDF3": [
        "CELL6",
        "CELL7"
    ]
    "UDF4": [
        "CELL8"
    ]
    "UDF5": [
        "CELL9"
    ]
    "UDF6": [
        "CELL10"
    ]
    "UDF7": [
        "CELL11"
    ]
}
"""

client = OpenAI(
    api_key="")


def call_gpt(prompt: str, conversation=None):
    if conversation is None:
        conversation = []

    conversation.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        messages=conversation,
        model="gpt-4o",
        temperature=0
    )

    conversation.append({"role": "assistant", "content": response.choices[0].message.content})

    return response.choices[0].message.content, conversation


def get_code_components(script: str, conversation):
    prompt = f"""
    Analyze the following Python code / Jupyter Notebook script and identify the key components such as imports, data loading, preprocessing, model training, evaluation, and visualization. Associate blocks of code with a key component.

    ```python
    {script}
    ```
    """
    format_prompt = "Format the response as a json object as the following:  {imports: str, data_loading: str, ...}"
    components, conversation = call_gpt(prompt + format_prompt, conversation)
    return (components, conversation)


def python_script_to_texera(python_script: str, do_print=False) -> tuple[str, str]:
    """
    Sends Python code to OpenAI to convert to a Texera workflow and mapping
    :param python_script: Python code to send to OpenAI
    :param do_print: flag that controls whether the program prints which step it is on
    :return: OpenAI workflow JSON and uni-directional mapping
    """
    conversation = []

    def append_response(prompt: str, conversation: list) -> tuple:
        response, conversation = call_gpt(prompt, conversation)
        return response, conversation

    def process_documentation_steps(conversation: list) -> list:
        documentation_steps = [
            texera_overview,
            tuple_documentation,
            table_documentation,
            operator_documentation,
            example_of_good_conversion,
            visualizer_documentation,
            udf_input_port_documentation
        ]

        for i, step in enumerate(documentation_steps):
            if do_print:
                print(f"\tProcessing documentation {i + 1} out of {len(documentation_steps)}...", end="")
            _, conversation = append_response(step, conversation)
            if do_print:
                print("DONE")

        return conversation

    def build_final_prompt(python_script: str) -> str:
        starter_prompt = ("Create multiple Texera UDF codes using the provided Python code."
                          "Number each UDF, starting at 1 and incrementing, by starting with a comment that states that UDF number.")
        using_only_class = ('Use the class and function names as shown in ProcessTupleOperator, '
                            'ProcessTableOperator, and ProcessBatchOperator. Do not change the class names, '
                            'function names, or input parameters. Use the ones that make sense and split the code '
                            'meaningfully as instructed.')
        use_starter_code = 'Use the starter code provided for Python UDFs.'
        use_documentation = (
            'Use the documentation of Table, Tuple, or Batch to work with parameters within Texera UDF. '
            'Do not import other libraries to define these types.')
        no_need_for_init = ('There is no need for an __init__ function. Assume all inputs are valid pandas DataFrames, '
                            'so do not use .to_pandas(), .to_dataframe(), etc. Do not load data from a file in the first UDF, assume '
                            'that the data is already given to you in the table parameter. '
                            'Ensure proper data flow between functions. Separate operators as if they will run in different files.')
        one_output_per_operator = (
            'Current UDF operators can only have one output. Build a dataframe to yield all necessary variables '
            'and data. Ensure proper data flow for each UDF and all information is yielded (including training '
            'and testing data) if subsequent UDFs need them.')
        ensure_imports = 'Ensure all necessary imports are included in each UDF code block.'
        separate_scripts = (
            'Each UDF operator should be in its own Python code block. Do not combine them into a single block. '
            'Ensure import statements cover all used functions and separate them as necessary.')
        python_to_convert = f"Here is the Python code. Return only the JSON formatted response, do not give any explanation. Convert following the instructions and examples given:\n{python_script}"

        return f"{starter_prompt}\n{using_only_class}\n{use_starter_code}\n{use_documentation}\n{no_need_for_init}\n{one_output_per_operator}\n{ensure_imports}\n{separate_scripts}\n{example_of_multiple_udf_conversion}\n{python_to_convert}"

    # Get code components and initial conversation
    if do_print:
        print("Getting code components from OpenAI...", end="")
    code_components, conversation = get_code_components(python_script, conversation)
    if do_print:
        print("DONE")

    # Process all documentation steps
    if do_print:
        print("Processing documentation steps...", end="")
    conversation = process_documentation_steps(conversation)
    if do_print:
        print("DONE")

    # Build and call the final prompt
    if do_print:
        print("Calling final udf prompt...", end="")
    final_prompt = build_final_prompt(python_script)
    if do_print:
        print("DONE")

    response, conversation = append_response(final_prompt, conversation)

    # Build bidirectional mapping
    if do_print:
        print("Generating mapping...", end="")
    mapping_prompt = f"{example_of_mapping}\nNow create a mapping for the UDFs and the original code. Link the code blocks marked by 'START CELL#' and 'END CELL#' with the numbered UDFs. The code between them should be equivalent. Multiple cells can be mapped to the same UDF if the code they contain are the same. There could be any number of cells and UDFs, so only create the correct number in the mapping. Only give the mapping."
    mapping, conversation = call_gpt(mapping_prompt, conversation)
    if do_print:
        print("DONE")

    if do_print:
        print("--------------------------------------- FINISHED ---------------------------------------")

    return response, mapping


def parse_workflow_and_mapping(open_ai_workflow, open_ai_mapping):
    """
    Converts the OpenAI workflow JSON into a Texera-compatible JSON and uni-directional mapping to bidirectional mapping
    :param open_ai_workflow: OpenAI workflow
    :param open_ai_mapping: uni-directional mapping
    :return: Texera JSON
    """
    udf_open_ai_response = json.loads(open_ai_workflow.strip("```json").strip("```").strip(), strict=False)

    csv_uuid = str(uuid4())

    workflow_json = \
        {
            "operators": [
                {
                    "operatorID": f"CSVFileScan-operator-{csv_uuid}",
                    "operatorType": "CSVFileScan",
                    "operatorVersion": "1fa249a9d55d4dcad36d93e093c2faed5c4434f0",
                    "operatorProperties": {
                        "fileEncoding": "UTF_8",
                        "customDelimiter": ",",
                        "hasHeader": True,
                        "fileName": ""
                    },
                    "inputPorts": [],
                    "outputPorts": [
                        {
                            "portID": "output-0",
                            "displayName": "",
                            "allowMultiInputs": False,
                            "isDynamicPort": False
                        }
                    ],
                    "showAdvanced": False,
                    "isDisabled": False,
                    "customDisplayName": "CSV File",
                    "dynamicInputPorts": False,
                    "dynamicOutputPorts": False
                }
            ],
            "operatorPositions": {
                f"CSVFileScan-operator-{csv_uuid}": {
                    "x": 0,
                    "y": 0
                }
            },
            "links": [],
            "groups": [],
            "commentBoxes": [],
            "settings": {
                "dataTransferBatchSize": 400
            }
        }

    udf_mapping_to_uuid = {}

    for i, (UDF_ID, UDF_code) in enumerate(udf_open_ai_response["code"].items(), start=1):
        udf_uuid = f"PythonUDFV2-operator-{str(uuid4())}"
        udf_mapping_to_uuid[UDF_ID] = udf_uuid
        udf_output_columns = [{"attributeName": attr, "attributeType": "binary"} for attr in
                              udf_open_ai_response["outputs"][UDF_ID]]

        # Add UDF to operators
        workflow_json["operators"].append(
            {
                "operatorID": f"{udf_uuid}",
                "operatorType": "PythonUDFV2",
                "operatorVersion": "3d69fdcedbb409b47162c4b55406c77e54abe416",
                "operatorProperties": {
                    "code": UDF_code,
                    "workers": 1,
                    "retainInputColumns": False,
                    "outputColumns": udf_output_columns
                },
                "inputPorts": [
                    {
                        "portID": "input-0",
                        "displayName": "",
                        "allowMultiInputs": True,
                        "isDynamicPort": False,
                        "dependencies": []
                    }
                ],
                "outputPorts": [
                    {
                        "portID": "output-0",
                        "displayName": "",
                        "allowMultiInputs": False,
                        "isDynamicPort": False
                    }
                ],
                "showAdvanced": False,
                "isDisabled": False,
                "customDisplayName": UDF_ID,
                "dynamicInputPorts": True,
                "dynamicOutputPorts": True
            }
        )

        # Add UDF to operatorPositions
        workflow_json["operatorPositions"][udf_uuid] = {"x": 140 * i, "y": 0}

    # Add links/edges
    for source, target in udf_open_ai_response["edges"]:
        workflow_json["links"].append(
            {
                "linkID": f"link-{str(uuid4())}",
                "source": {
                    "operatorID": udf_mapping_to_uuid[source],
                    "portID": "output-0"
                },
                "target": {
                    "operatorID": udf_mapping_to_uuid[target],
                    "portID": "input-0"
                }
            }
        )

    # Parses the mapping
    mapping = json.loads(open_ai_mapping.strip("```json\n").strip("```"))

    udf_to_cell = {}
    cell_to_udf = {}
    for udf, cells in mapping.items():
        udf_uuid = udf_mapping_to_uuid[udf]
        udf_to_cell[udf_uuid] = cells
        for cell in cells:
            if cell not in cell_to_udf:
                cell_to_udf[cell] = [udf_uuid]
            else:
                cell_to_udf[cell].append(udf_uuid)

    combined_mapping = {
        "operator_to_cell": udf_to_cell,
        "cell_to_operator": cell_to_udf
    }

    return workflow_json, combined_mapping


def notebook_to_texera(do_print=False):
    """
    Calls OpenAI with Texera documentation to generate and parse a workflow JSON and mapping JSON
    :param notebook_path: path to the Jupyter notebook
    :param do_print: flag that controls whether the program prints which step it is on
    :return: a JSON representing the workflow, and a JSON representing the mapping
    """
    # Load the notebook
    while not NOTEBOOK_SAVED:
        sleep(2)
    notebook_path = os.path.join(NOTEBOOK_PATH, NOTEBOOK_NAME)
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook_file = nbf.read(f, as_version=4)

    # Extract code cells and maintain separation
    code_cells = [cell for cell in notebook_file.cells if cell.cell_type == 'code']

    # Join the code into one string
    notebook_string = "\n\n".join(
        f"# START {cell['metadata']['uuid']}\n"
        f"{cell['source']}\n"
        f"# END {cell['metadata']['uuid']}"
        for cell in code_cells
    )

    open_ai_workflow, open_ai_mapping = python_script_to_texera(notebook_string, do_print)

    final_workflow, final_mapping = parse_workflow_and_mapping(open_ai_workflow, open_ai_mapping)

    return jsonify({"workflow": final_workflow, "mapping": final_mapping}), 200


@app.route('/set_notebook', methods=['POST'])
def set_notebook():
    try:
        data = request.json
        global NOTEBOOK_NAME
        NOTEBOOK_NAME = data.get('notebookName', 'example.ipynb')
        notebook_data = data.get('notebookData')

        for cell in notebook_data["cells"]:
            if 'metadata' not in cell:
                cell['metadata'] = {}
            cell['metadata']['uuid'] = str(uuid4())

        if not notebook_data:
            return jsonify({"error": "Notebook data is required"}), 400

        # Save the notebook JSON file
        notebook_file_path = os.path.join(NOTEBOOK_PATH, NOTEBOOK_NAME)
        with open(notebook_file_path, 'w', encoding='utf-8') as f:
            json.dump(notebook_data, f, indent=4)

        global NOTEBOOK_SAVED
        NOTEBOOK_SAVED = True

        return jsonify({
            "message": "Notebook saved successfully",
            "notebookPath": notebook_file_path
        })

    except Exception as e:
        print(f"Unexpected server error: {e}")
        return jsonify({"error": f"Unexpected server error: {str(e)}"}), 500


@app.route('/get_openai_response', methods=['POST'])
def get_openai_response():
    return notebook_to_texera(do_print=True)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
