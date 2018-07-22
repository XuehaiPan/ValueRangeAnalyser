from math import inf, isinf, isfinite, isnan, floor
from typing import Type, Union, Sequence, List, Tuple, Callable


__all__: List[str] = ['ValueRange', 'EmptySet', 'IntegerNumberSet', 'RealNumberSet', 'dtypeFromString']

BasicValueRange: Type = type('BasicValueRange', (object,), dict())
ValueRange: Type = type('ValueRange', (object,), dict())


def dtypeFromString(dtype: str) -> Union[Type[int], Type[float]]:
    if dtype == 'int':
        return int
    elif dtype == 'float':
        return float
    else:
        raise ValueError


class BasicValueRange(object):
    def __init__(self, lower: Union[int, float] = None, upper: Union[int, float] = None,
                 dtype: Union[str, Type[int], Type[float]] = float) -> None:
        if isinstance(dtype, str):
            dtype: Union[Type[int], Type[float]] = dtypeFromString(dtype = dtype)
        if dtype != int and dtype != float:
            raise ValueError
        self.__dtype: Type[dtype] = dtype
        try:
            if dtype == int:
                try:
                    self.__lower: int = floor(lower)
                except OverflowError:
                    self.__lower: float = lower
                try:
                    self.__upper: int = floor(upper)
                except OverflowError:
                    self.__upper: float = upper
            else:
                self.__lower: float = float(lower)
                self.__upper: float = float(upper)
            if self.lower > self.upper or isnan(lower) or isnan(upper) \
                    or (isinf(lower) and isinf(upper) and lower == upper):
                self.__lower, self.__upper = None, None
            elif self.dtype == int:
                if isfinite(lower) and abs(self.lower + 1 - round(lower)) < 1E-6:
                    self.__lower: int = round(lower)
                if isfinite(upper) and abs(self.upper + 1 - round(upper)) < 1E-6:
                    self.__upper: int = round(upper)
        except TypeError:
            self.__lower, self.__upper = None, None
    
    @property
    def dtype(self) -> Union[Type[int], Type[float]]:
        return self.__dtype
    
    @property
    def lower(self) -> Union[int, float]:
        return self.__lower
    
    @property
    def upper(self) -> Union[int, float]:
        return self.__upper
    
    @property
    def bound(self) -> Tuple[Union[int, float], Union[int, float]]:
        return self.lower, self.upper
    
    @bound.setter
    def bound(self, newBound: Tuple[Union[int, float], Union[int, float]]) -> None:
        self.__lower, self.__upper = BasicValueRange(lower = newBound[0], upper = newBound[1], dtype = self.dtype).bound
    
    def copy(self) -> BasicValueRange:
        return BasicValueRange(lower = self.lower, upper = self.upper, dtype = self.dtype)
    
    def isEmptySet(self) -> bool:
        if self.lower is None or self.upper is None or self.lower > self.upper:
            self.__lower, self.__upper = None, None
            return True
        else:
            return False
    
    def isOverlapping(self, other: BasicValueRange) -> bool:
        try:
            return self.lower <= other.lower <= self.upper or self.lower <= other.upper <= self.upper
        except TypeError:
            return False
    
    def isDisjoint(self, other: BasicValueRange) -> bool:
        return not self.isOverlapping(other = other)
    
    def isSubset(self, other: BasicValueRange) -> bool:
        if self.isEmptySet():
            return True
        else:
            try:
                return other.lower <= self.lower and self.upper <= other.upper
            except ValueError:
                return False
    
    def isSuperset(self, other: BasicValueRange) -> bool:
        return other.isSubset(other = self)
    
    def asDtype(self, dtype: Union[str, Type[int], Type[float]]) -> BasicValueRange:
        return BasicValueRange(lower = self.lower, upper = self.upper, dtype = dtype)
    
    def asInt(self) -> BasicValueRange:
        return self.asDtype(dtype = int)
    
    def asFloat(self) -> BasicValueRange:
        return self.asDtype(dtype = float)
    
    def __len__(self) -> Union[int, float]:
        if self.isEmptySet():
            return 0
        else:
            return self.upper - self.lower
    
    def __bool__(self) -> bool:
        return self.isEmptySet()
    
    def __eq__(self, other: BasicValueRange) -> bool:
        return self.isSubset(other = other) and other.isSubset(other = self)
    
    def __ne__(self, other: BasicValueRange) -> bool:
        return not self.__eq__(other = other)
    
    def __contains__(self, value: Union[int, float]) -> bool:
        try:
            return self.lower <= value <= self.upper
        except TypeError:
            return False
    
    def __str__(self) -> str:
        if self.isEmptySet():
            return 'EmptySet'
        else:
            if self.lower == self.upper:
                return '{{ {} }}'.format(self.lower)
            elif isinf(self.lower) and isinf(self.upper):
                return '(-inf, +inf)'
            elif isinf(self.lower):
                return '(-inf, {}]'.format(self.upper)
            elif isinf(self.upper):
                return '[{}, +inf)'.format(self.lower)
            else:
                return '[{}, {}]'.format(self.lower, self.upper)
    
    __repr__: Callable[[BasicValueRange], str] = __str__


class ValueRange(object):
    def __init__(self, *args, **kwargs) -> None:
        if len(args) + len(kwargs) != 1:
            self.__init__(valueRange = BasicValueRange(*args, **kwargs))
            return
        valueRanges: List[BasicValueRange] = None
        if len(args) == 1:
            if isinstance(args[0], ValueRange):
                valueRanges: List[BasicValueRange] = args[0].valueRanges
            elif isinstance(args[0], BasicValueRange):
                valueRanges: List[BasicValueRange] = [args[0]]
            elif isinstance(args[0], Sequence):
                valueRanges: List[BasicValueRange] = args[0]
        elif len(kwargs) == 1:
            try:
                valueRanges: List[BasicValueRange] = kwargs['valueRanges']
            except KeyError:
                valueRanges: List[BasicValueRange] = [kwargs['valueRange']]
        valueRanges = list(filter(lambda s: not s.isEmptySet(), valueRanges))
        dtype: Type[int] = int
        if any(valueRange.dtype == float for valueRange in valueRanges):
            dtype: Type[float] = float
        self.__dtype: Type[dtype] = dtype
        try:
            try:
                self.__lower: dtype = dtype(min(valueRange.lower for valueRange in valueRanges))
            except OverflowError:
                self.__lower: float = -inf
            try:
                self.__upper: dtype = dtype(max(valueRange.upper for valueRange in valueRanges))
            except OverflowError:
                self.__upper: float = +inf
        except ValueError:
            self.__lower, self.__upper = None, None
        if dtype == float:
            self.__valueRanges: List[BasicValueRange] = list(map(BasicValueRange.asFloat, valueRanges))
        else:
            self.__valueRanges: List[BasicValueRange] = list(map(BasicValueRange.copy, valueRanges))
        self.reduce()
    
    @property
    def dtype(self) -> Union[Type[int], Type[float]]:
        return self.__dtype
    
    @property
    def lower(self) -> Union[int, float]:
        return self.__lower
    
    @property
    def upper(self) -> Union[int, float]:
        return self.__upper
    
    @property
    def bound(self) -> Union[Tuple[int, int], Tuple[float, float]]:
        return self.lower, self.upper
    
    @property
    def valueRanges(self) -> List[BasicValueRange]:
        return self.__valueRanges
    
    def reduce(self) -> None:
        if not self.isEmptySet():
            self.__valueRanges: List[BasicValueRange] = list(valueRange.asDtype(dtype = self.dtype) for valueRange in self.valueRanges)
            self.valueRanges.sort(key = lambda valueRange: valueRange.lower)
            valueRanges: List[BasicValueRange] = [self.valueRanges[0]]
            for valueRange in self.valueRanges[1:]:
                if valueRanges[-1].isOverlapping(other = valueRange):
                    if valueRange.isSubset(other = valueRanges[-1]):
                        continue
                    else:
                        valueRanges[-1].bound = valueRanges[-1].lower, valueRange.upper
                elif self.dtype == int and valueRanges[-1].upper + 1 == valueRange.lower:
                    valueRanges[-1].bound = valueRanges[-1].lower, valueRange.upper
                else:
                    valueRanges.append(valueRange)
            self.__valueRanges: List[BasicValueRange] = valueRanges
            self.__lower: self.dtype = valueRanges[0].lower
            self.__upper: self.dtype = valueRanges[-1].upper
    
    def copy(self) -> ValueRange:
        return ValueRange(valueRanges = self.valueRanges)
    
    def isEmptySet(self) -> bool:
        if len(self.valueRanges) == 0 or self.lower is None or self.upper is None:
            self.valueRanges.clear()
            self.__lower, self.__upper = None, None
            return True
        else:
            return False
    
    def isIntegerNumberSet(self) -> bool:
        return len(self.valueRanges) == 1 and isinf(self.lower) and isinf(self.upper) and self.dtype == int
    
    def isRealNumberSet(self) -> bool:
        return len(self.valueRanges) == 1 and isinf(self.lower) and isinf(self.upper) and self.dtype == float
    
    def addValueRange(self, subset: BasicValueRange) -> None:
        if not subset.isEmptySet():
            self.valueRanges.append(subset)
            if subset.dtype == float and self.dtype == int:
                self.__dtype: Type[float] = float
                self.__valueRanges: List[BasicValueRange] = list(map(BasicValueRange.asFloat, self.valueRanges))
            self.reduce()
    
    def isOverlapping(self, other: ValueRange) -> bool:
        return any(any(map(valueRange.isOverlapping, other.valueRanges)) for valueRange in self.valueRanges)
    
    def isDisjoint(self, other: ValueRange) -> bool:
        return not self.isOverlapping(other = other)
    
    def isSubset(self, other: ValueRange) -> bool:
        return all(any(map(valueRange.isSubset, other.valueRanges)) for valueRange in self.valueRanges)
    
    def isSuperset(self, other: ValueRange) -> ValueRange:
        return other.isSubset(other = self)
    
    def union(self, other: ValueRange) -> ValueRange:
        return ValueRange(valueRanges = self.valueRanges + other.valueRanges)
    
    def difference(self, other: ValueRange) -> ValueRange:
        diffValueRanges: List[BasicValueRange] = list()
        dtype: Type[float] = float
        if (self.dtype == int or self.isRealNumberSet()) and other.dtype == int:
            dtype: Type[int] = int
        valueRanges: List[BasicValueRange] = list(valueRange.asDtype(dtype = dtype) for valueRange in self.valueRanges)
        for valueRange in valueRanges:
            if any(map(valueRange.isSubset, other.valueRanges)):
                continue
            for otherValueRange in other.valueRanges:
                if valueRange.isOverlapping(other = otherValueRange):
                    if valueRange.isSuperset(other = otherValueRange):
                        if dtype == int:
                            valueRanges.append(BasicValueRange(lower = otherValueRange.upper + 1,
                                                               upper = valueRange.upper,
                                                               dtype = dtype))
                            valueRange.bound = valueRange.lower, otherValueRange.lower - 1
                        else:
                            valueRanges.append(BasicValueRange(lower = otherValueRange.upper,
                                                               upper = valueRange.upper,
                                                               dtype = dtype))
                            valueRange.bound = valueRange.lower, otherValueRange.lower
                    else:
                        if valueRange.lower in otherValueRange:
                            if dtype == int:
                                valueRange.bound = otherValueRange.upper + 1, valueRange.upper
                            else:
                                valueRange.bound = otherValueRange.upper, valueRange.upper
                        if valueRange.upper in otherValueRange:
                            if dtype == int:
                                valueRange.bound = valueRange.lower, otherValueRange.lower - 1
                            else:
                                valueRange.bound = valueRange.lower, otherValueRange.lower
            diffValueRanges.append(valueRange)
        return ValueRange(valueRanges = diffValueRanges)
    
    def intersection(self, other: ValueRange) -> ValueRange:
        return self.difference(other = self.difference(other = other))
    
    def symmetricDifference(self, other: ValueRange) -> ValueRange:
        return self.difference(other = other).union(other = other.difference(other = self))
    
    def update(self, other: ValueRange) -> None:
        if not other.isEmptySet():
            self.valueRanges.extend(other.valueRanges)
            if other.dtype == float and self.dtype == int:
                self.__dtype: Type[float] = float
                self.__valueRanges: List[BasicValueRange] = list(map(BasicValueRange.asFloat, self.valueRanges))
            self.reduce()
    
    def asDtype(self, dtype: Union[str, Type[int], Type[float]]) -> ValueRange:
        return ValueRange(valueRanges = list(map(lambda valueRange: valueRange.asDtype(dtype = dtype), self.valueRanges)))
    
    def __iadd__(self, other: Union[int, float, BasicValueRange, ValueRange]) -> None:
        other: ValueRange = ValueRange.asValueRange(value = other)
        sumValueRanges: List[BasicValueRange] = list()
        dtype: Type[float] = float
        if self.dtype == int and other.dtype == int:
            dtype: Type[int] = int
        for valueRange in self.valueRanges:
            for otherValueRange in other.valueRanges:
                sumValueRanges.append(BasicValueRange(lower = valueRange.lower + otherValueRange.lower,
                                                      upper = valueRange.upper + otherValueRange.upper,
                                                      dtype = dtype))
        self.__dtype: Union[Type[int], Type[float]] = dtype
        self.__valueRanges: List[BasicValueRange] = sumValueRanges
        self.reduce()
    
    def __add__(self, other: Union[int, float, BasicValueRange, ValueRange]) -> ValueRange:
        res: ValueRange = self.copy()
        res.__iadd__(other = other)
        return res
    
    def __isub__(self, other: Union[int, float, BasicValueRange, ValueRange]) -> None:
        self.__iadd__(other = -other)
    
    def __sub__(self, other: Union[int, float, BasicValueRange, ValueRange]) -> ValueRange:
        return self.__add__(other = -other)
    
    def __imul__(self, other: Union[int, float, BasicValueRange, ValueRange]) -> None:
        other: ValueRange = ValueRange.asValueRange(value = other)
        prodValueRanges: List[BasicValueRange] = list()
        dtype: Type[float] = float
        if self.dtype == int and other.dtype == int:
            dtype: Type[int] = int
        for valueRange in self.valueRanges:
            for otherValueRange in other.valueRanges:
                bounds: List[Union[int, float]] = [valueRange.lower * otherValueRange.lower,
                                                   valueRange.upper * otherValueRange.lower,
                                                   valueRange.lower * otherValueRange.upper,
                                                   valueRange.upper * otherValueRange.upper]
                prodValueRanges.append(BasicValueRange(lower = min(bounds), upper = max(bounds), dtype = dtype))
        self.__dtype: Union[Type[int], Type[float]] = dtype
        self.__valueRanges: List[BasicValueRange] = prodValueRanges
        self.reduce()
    
    def __mul__(self, other: Union[int, float, BasicValueRange, ValueRange]) -> ValueRange:
        res: ValueRange = self.copy()
        res.__imul__(other = other)
        return res
    
    def __itruediv__(self, other: Union[int, float, BasicValueRange, ValueRange]) -> None:
        other: ValueRange = ValueRange.asValueRange(value = other)
        dtype: Type[float] = float
        if self.dtype == int and other.dtype == int:
            other: ValueRange = other.difference(other = ValueRange(lower = 0, upper = 0, dtype = int))
            divValueRanges: List[BasicValueRange] = list()
            for valueRange in self.valueRanges:
                for otherValueRange in other.valueRanges:
                    bounds: List[int] = [valueRange.lower // otherValueRange.lower,
                                         valueRange.upper // otherValueRange.lower,
                                         valueRange.lower // otherValueRange.upper,
                                         valueRange.upper // otherValueRange.upper]
                    divValueRanges.append(BasicValueRange(lower = min(bounds), upper = max(bounds), dtype = dtype))
            self.__valueRanges: List[BasicValueRange] = divValueRanges
        else:
            recipValueRanges: List[BasicValueRange] = list()
            for otherValueRange in other.valueRanges:
                if 0.0 in otherValueRange:
                    if otherValueRange.lower < 0:
                        recipValueRanges.append(BasicValueRange(lower = -inf,
                                                                upper = 1.0 / otherValueRange.lower,
                                                                dtype = float))
                    if otherValueRange.upper > 0:
                        recipValueRanges.append(BasicValueRange(lower = 1.0 / otherValueRange.upper,
                                                                upper = +inf,
                                                                dtype = float))
                else:
                    recipValueRanges.append(BasicValueRange(lower = 1.0 / otherValueRange.upper,
                                                            upper = 1.0 / otherValueRange.lower,
                                                            dtype = float))
            self.__imul__(other = ValueRange(valueRanges = recipValueRanges))
        self.reduce()
    
    def __truediv__(self, other: Union[int, float, BasicValueRange, ValueRange]) -> ValueRange:
        res: ValueRange = self.copy()
        res.__itruediv__(other = other)
        return res
    
    def __neg__(self) -> ValueRange:
        res: ValueRange = self.copy()
        for valueRange in res.valueRanges:
            valueRange.bound = (-valueRange.upper, -valueRange.lower)
        return res
    
    def __bool__(self) -> bool:
        return self.isEmptySet()
    
    def __eq__(self, other: Union[int, float, BasicValueRange, ValueRange]) -> bool:
        other: ValueRange = ValueRange.asValueRange(value = other)
        return self.isSubset(other = other) and other.isSubset(other = self)
    
    def __ne__(self, other: Union[int, float, BasicValueRange, ValueRange]) -> bool:
        return not self.__eq__(other = other)
    
    def __contains__(self, value: Union[int, float]) -> bool:
        return any(value in valueRange for valueRange in self.valueRanges)
    
    def __str__(self) -> str:
        if self.isEmptySet():
            return 'EmptySet'
        else:
            return ' U '.join(map(str, self.valueRanges))
    
    __repr__: Callable[[ValueRange], str] = __str__
    
    @staticmethod
    def asValueRange(value: Union[int, float, BasicValueRange, ValueRange]) -> ValueRange:
        if not isinstance(value, ValueRange):
            if isinstance(value, BasicValueRange):
                return ValueRange(valueRange = value)
            if isinstance(value, int):
                return ValueRange(lower = value, upper = value, dtype = int)
            elif isinstance(value, float):
                return ValueRange(lower = value, upper = value, dtype = float)
            else:
                raise ValueError
        else:
            return value


EmptySet: ValueRange = ValueRange(lower = None, upper = None, dtype = int)
IntegerNumberSet: ValueRange = ValueRange(lower = -inf, upper = +inf, dtype = int)
RealNumberSet: ValueRange = ValueRange(lower = -inf, upper = +inf, dtype = float)
