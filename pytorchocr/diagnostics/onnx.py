def static_onnx_input_shape(session):
    shape = session.get_inputs()[0].shape
    if len(shape) != 4 or not all(isinstance(value, int) for value in shape[1:]):
        raise ValueError(f"ONNX must have static channel/height/width: {shape}")
    return shape


def resolve_onnx_input_shape(session, image_shape):
    """Resolve dynamic ONNX spatial dimensions with an explicit OCR shape."""
    shape = list(session.get_inputs()[0].shape)
    image_shape = tuple(image_shape)
    if len(shape) != 4 or len(image_shape) != 3:
        raise ValueError(
            f"Expected ONNX NCHW input and a CHW image shape: {shape}, {image_shape}"
        )
    if any(not isinstance(value, int) or value <= 0 for value in image_shape):
        raise ValueError(f"Image shape must contain positive integers: {image_shape}")
    for axis, requested in enumerate(image_shape, start=1):
        declared = shape[axis]
        if isinstance(declared, int) and declared != requested:
            raise ValueError(
                f"ONNX input axis {axis} is {declared}, requested {requested}."
            )
        shape[axis] = requested
    return shape
