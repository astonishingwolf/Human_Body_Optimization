import math
import torch
from typing import Any, List, Callable, Tuple, Union, NamedTuple, Optional
from einops import rearrange, repeat


class EasyDict(dict):
    """Convenience class that behaves like a dict but allows access with the attribute syntax."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        del self[name]

    @staticmethod
    def from_dict(d: dict) -> None:
        """Update the dict with the given dict."""
        obj = EasyDict()
        for k, v in d.items():
            if isinstance(v, dict):
                obj[k] = EasyDict.from_dict(v)
            elif isinstance(v, list) and isinstance(v[0], dict):
                obj[k] = [EasyDict.from_dict(v_) for v_ in v]
            else:
                obj[k] = v
        return obj


def _is_tensor(v: Any) -> bool:
    """Check if the value is a tensor."""
    return isinstance(v, torch.Tensor) or isinstance(v, TensorDict)


class TensorDict(EasyDict):
    """
    A dictionary subclass specialized for handling nested structures of PyTorch Tensors.
    
    It allows for:
    1.  Recursive device movement (.to(device)).
    2.  Batch manipulation (split, stack, cat, rearrange).
    3.  Easy access to tensor and non-tensor components.
    4.  Seamless integration with PyTorch logic requiring dict-like inputs.
    """
    @staticmethod
    def from_dict(d: dict) -> 'TensorDict':
        """Create a TensorDict from a nested dict."""
        obj = TensorDict()
        for k, v in d.items():
            if isinstance(v, dict):
                obj[k] = TensorDict.from_dict(v)
            else:
                obj[k] = v
        return obj

    def valid_dic(self) -> dict:
        """Return the valid dict that is not None."""
        return {k: v for k, v in self.items() if v is not None}

    def tensor_keys(self) -> List[str]:
        """Return the tensor keys."""
        return [k for k, v in self.items() if _is_tensor(v)]

    def nontensor_keys(self) -> List[str]:
        """Return the non-tensor keys."""
        return [k for k, v in self.items() if not _is_tensor(v)]

    def tensor_dic(self) -> dict:
        """Return the tensor components in the dict."""
        return {k: v for k, v in self.items() if _is_tensor(v)}

    def nontensor_dic(self) -> dict:
        """Return the non-tensor components in the dict."""
        return {k: v for k, v in self.items() if not _is_tensor(v)}

    def apply(self, func: Callable, *args, **kwargs) -> 'TensorDict':
        """Apply a function to the tensor components."""
        for k, v in self.tensor_dic().items():
            self[k] = func(v, *args, **kwargs)
        return self
    
    def to(self, device: torch.device, dtype: Optional[torch.dtype] = None) -> 'TensorDict':
        """Move all tensors to the device."""
        if dtype is None:
            return self.apply(lambda x: x.to(device))
        else:
            return self.apply(lambda x: x.to(device, dtype=dtype))

    def split(self, chunk_size: int = 1, dim: int = 0, squeeze: bool = True) -> List['TensorDict']:
        """Split the tensor components on given dim.
        Args:
            chunk_size: the size of each chunk.
            squeeze: whether to squeeze when chunk_size = 1
        """

        """
        chunks = None
        for k, v in self.tensor_dic().items():
            n_chunk = math.ceil(float(v.shape[dim]) / chunk_size)
            if chunks is None:
                chunks = [self.nontensor_dic() for _ in range(n_chunk)]
            for i in range(n_chunk):
                if dim == 0:
                    chunks[i][k] = v[i * chunk_size : (i + 1) * chunk_size]
                elif dim == 1:
                    chunks[i][k] = v[:, i * chunk_size : (i + 1) * chunk_size]
                else:
                    raise ValueError(f"Dimension {dim} not supported")
                if squeeze:
                    chunks[i][k] = chunks[i][k].squeeze(dim=dim)
        assert chunks is not None, "TensorDict: No tensor components to split"
        """

        tensor_items = list(self.tensor_dic().items())
        first_key, first_val = tensor_items[0]
        n_chunk = math.ceil(float(first_val.shape[dim]) / chunk_size)
        
        chunks = [self.nontensor_dic() for _ in range(n_chunk)]
        
        for k, v in tensor_items:
            for i in range(n_chunk):
                if dim == 0:
                    chunks[i][k] = v[i * chunk_size : (i + 1) * chunk_size]
                elif dim == 1:
                    chunks[i][k] = v[:, i * chunk_size : (i + 1) * chunk_size]
                else:
                    raise ValueError(f"Dimension {dim} not supported")
                if squeeze:
                    chunks[i][k] = chunks[i][k].squeeze(dim=dim)
        
        return [self.__class__(ch) for ch in chunks]

    def rearrange(self, *kargs, **kwargs):
        return self.apply(rearrange, *kargs, **kwargs)
    
    def repeat(self, *kargs, **kwargs):
        return self.apply(repeat, *kargs, **kwargs)

    def detach(self):
        """Detach all tensors."""
        return self.apply(lambda x: getattr(x, 'detach')())

    def clone(self):
        """Detach all tensors."""
        dic = self.nontensor_dic()
        dic.update({k: v.clone() for k, v in self.tensor_dic().items()})
        return self.__class__(dic)

    def squeeze(self, dim=0):
        """Squeeze all tensors."""
        func = lambda x, dim: getattr(x, 'squeeze')(dim=dim)
        return self.apply(func, dim=dim)
    
    def unsqueeze(self, dim=0):
        """Unsqueeze all tensors."""
        func = lambda x, dim: getattr(x, 'unsqueeze')(dim=dim)
        return self.apply(func, dim=dim)

    def requires_grad_(self, requires_grad=True):
        """Set requires_grad for all tensors."""
        return self.apply(lambda x: x.requires_grad_(requires_grad))

    def slice_between(self, bg, st, dim=0):
        """Slice all tensors between bg and st."""
        dic = self.nontensor_dic()
        for k, v in self.tensor_dic().items():
            if isinstance(v, torch.Tensor):
                if dim == 0:
                    dic[k] = v[bg:st]
                elif dim == 1:
                    dic[k] = v[:, bg:st]
            else:
                dic[k] = v
        return self.__class__(dic)

    def slice_indice(self, indice, dim=0):
        """Slice all tensors between bg and st."""
        dic = self.nontensor_dic()
        dic.update({k: v[indice] if dim == 0 else v[:, indice]
                    for k, v in self.tensor_dic().items()})
        return self.__class__(dic)
    
    def mul_(self, val):
        """Multiply all tensors by a value."""
        if isinstance(val, TensorDict):
            for k, v in val.tensor_dic().items():
                self[k] = self[k] * v
        else:
            self.apply(lambda x: x * val)
        return self
    
    def add_(self, val):
        """Add a value to all tensors."""
        if isinstance(val, TensorDict):
            for k, v in val.tensor_dic().items():
                self[k] = self[k] + v
        else:
            self.apply(lambda x: x + val)
        return self

    @staticmethod
    def list_apply(td_list: List['TensorDict'], func: Callable, *args, **kwargs):
        """Apply a function to the list of tensor dicts."""
        dic = td_list[0].nontensor_dic()
        dic.update({k: func([td[k] for td in td_list], *args, **kwargs)
                    for k in td_list[0].tensor_keys()})
        return TensorDict(dic)

    @staticmethod
    def cat(td_list: List['TensorDict'], dim: int = 0):
        """Concatenate the list of tensors."""
        dic = td_list[0].nontensor_dic()
        self_type = type(td_list[0])
        for k in td_list[0].tensor_keys():
            if isinstance(td_list[0][k], self_type):
                dic[k] = self_type.cat([td[k] for td in td_list], dim)
            else:
                dic[k] = torch.cat([td[k] for td in td_list], dim)
        return self_type(dic)

    @staticmethod
    def stack(td_list: List['TensorDict'], dim: int = 0):
        """Stack the list of tensors."""
        dic = td_list[0].nontensor_dic()
        self_type = type(td_list[0])
        for k in td_list[0].tensor_keys():
            if isinstance(td_list[0][k], self_type):
                dic[k] = self_type.stack([td[k] for td in td_list], dim)
            else:
                dic[k] = torch.stack([td[k] for td in td_list], dim)
        return self_type(dic)
        