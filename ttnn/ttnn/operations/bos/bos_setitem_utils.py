from typing import Any, List, Tuple
import ttnn

def get_slice_tuple(slices: Tuple[Any, ...], shape: Tuple[int, ...]) -> List[slice]:
    """
    Converts a PyTorch-style indexing tuple into the arguments needed for tensor indexing.
    """
    # Gather some basic info
    input_rank = len(shape)

    # 1) Normalize slices into a tuple.
    #    e.g. if user wrote a[3], slices is just int(3), so wrap it into (3,).
    #    or if user wrote a[2:5], slices is slice(2,5).
    if isinstance(slices, (int, slice, type(...), ttnn.Tensor)):
        slices = (slices,)
    else:
        # ensure it's a tuple in case user wrote something like a[1, 2:5, ...]
        slices = tuple(slices)

    # 2) Expand any bare Ellipsis into enough slice(None) to fill out to input_rank.
    #    But we have to do this carefully in the presence of other slices.
    #    We'll do it in two passes:
    #      - first copy slices to a “normalized_slices”, remembering where Ellipsis is
    #      - then replace the Ellipsis with however many slice(None) are needed
    normalized_slices = []
    ellipsis_found = False
    for s in slices:
        if s is Ellipsis:
            if ellipsis_found:
                raise ValueError("Only one ellipsis ('...') is allowed in a slice.")
            # We'll deal with actually expanding it after this loop.
            ellipsis_found = True
            normalized_slices.append(Ellipsis)
        else:
            normalized_slices.append(s)

    # If there's exactly one Ellipsis, expand it
    if ellipsis_found:
        ellipsis_index = normalized_slices.index(Ellipsis)
        # Number of slices ignoring the Ellipsis
        num_slices_no_ellipsis = len(normalized_slices) - 1
        # How many dimensions are “missing”
        num_missing = input_rank - num_slices_no_ellipsis
        if num_missing < 0:
            raise IndexError(f"Too many indices for tensor of dimension {input_rank}")

        # Remove the Ellipsis placeholder
        del normalized_slices[ellipsis_index]
        # Insert slice(None) for however many dims are missing
        for _ in range(num_missing):
            normalized_slices.insert(ellipsis_index, slice(None, None, None))

    # If there was no Ellipsis and we still have fewer slices than rank, pad with slice(None)
    while len(normalized_slices) < input_rank:
        normalized_slices.append(slice(None, None, None))

    # Now if we have more slices than the rank, that’s an error
    if len(normalized_slices) > input_rank:
        raise IndexError(f"Too many indices for tensor of dimension {input_rank}")

    # 3) Convert everything into slice objects (including integer indices),
    #    and record which dimensions we’ll need to squeeze out (integer-indexed dims).
    final_slices = []
    singled_out_dims = []  # dims where user gave an integer index
    index_tensors = None  # TODO: not supported
    index_dims = []

    for dim_idx, s in enumerate(normalized_slices):
        if isinstance(s, int):
            # Negative index => convert as in python: s + size if s < 0
            idx = s if s >= 0 else (s + shape[dim_idx])
            if not 0 <= idx < shape[dim_idx]:
                raise IndexError(
                    f"Index {s} (converted to {idx}) is out of bounds "
                    f"for dimension {dim_idx} of size {shape[dim_idx]}"
                )
            final_slices.append(slice(idx, idx + 1, 1))
            singled_out_dims.append(dim_idx)
        elif isinstance(s, slice):
            # We mimic Python negative slicing for start/stop
            start, stop, step = s.start, s.stop, s.step

            # default values
            if start is None:
                start = 0
            if stop is None:
                stop = shape[dim_idx]
            if step is None:
                step = 1

            final_slices.append(slice(start, stop, step))
        elif isinstance(s, ttnn.Tensor):
            final_slices.append(slice(0, shape[dim_idx], 1))
            if index_tensors is None:
                index_tensors = []
            index_tensors.append(s)
            index_dims.append(dim_idx)
        else:
            raise TypeError(f"Invalid slice type: {s}")

    # 4) Prepare the lists for ttnn.slice
    slice_start = []
    slice_end = []
    slice_step = []

    for dim_idx, sl in enumerate(final_slices):
        # No further negative indexing needed: we already converted above
        slice_start.append(sl.start)
        slice_end.append(sl.stop)
        slice_step.append(sl.step)

    return slice_start, slice_end, slice_step, index_tensors, index_dims


@ttnn.register_python_operation(
    name="ttnn.Tensor.__setitem__",
    is_method=True,
)
def ttnn_tensor_setitem(input_tensor: ttnn.Tensor, slices, value: ttnn.Tensor) -> ttnn.Tensor:
    # Setitem device operation
    assert isinstance(input_tensor, ttnn.Tensor), "ttnn.Tensor.__setitem__: input tensor must be a ttnn.Tensor!"
    assert isinstance(value, ttnn.Tensor), "ttnn.Tensor.__setitem__: value must be a ttnn.Tensor!"
    assert ttnn.is_tensor_storage_on_device(input_tensor), "ttnn.Tensor.__setitem__: input tensor must be on device!"
    assert ttnn.is_tensor_storage_on_device(value), "ttnn.Tensor.__setitem__: value tensor must be on device!"

    # 1) Normalize slices into a tuple.
    #    e.g. if user wrote a[3], slices is just int(3), so wrap it into (3,).
    #    or if user wrote a[2:5], slices is slice(2,5).
    if isinstance(slices, (int, slice, type(...), ttnn.Tensor)):
        slices = (slices,)
    else:
        # ensure it's a tuple in case user wrote something like a[1, 2:5, ...]
        slices = tuple(slices)

    # Remove None slices
    modified_slices = [s for s in slices if s is not None]

    # Convert slices to tuples
    begins, ends, steps, index_tensors, index_dims = get_slice_tuple(modified_slices, input_tensor.shape)
    # print(f"begins={begins}, ends={ends}, steps={steps}, index_tensors={index_tensors}, index_dims={index_dims}")

    ttnn.bos_setitem(
        input=input_tensor,
        value=value,
        begins=begins,
        ends=ends,
        steps=steps,
        index_tensors=index_tensors,
        index_dims=index_dims,
    )

# Monkey-patch the __setitem__ method to ttnn.Tensor
ttnn.Tensor.__setitem__ = lambda self, *args, **kwargs: ttnn_tensor_setitem(self, *args, **kwargs)
