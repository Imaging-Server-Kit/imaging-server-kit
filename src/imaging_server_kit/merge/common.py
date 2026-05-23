

def _get_slices_with_channel(cmin_rounded, cmax_rounded, channel_axis):
    """Convenience function to get the slices, accounting for the channel axis."""
    slices = tuple(
        [slice(cmin, cmax) for cmin, cmax in zip(cmin_rounded, cmax_rounded)]
    )

    if channel_axis is not None:
        slices_with_channel = (
            slices[:channel_axis] + (slice(None),) + slices[channel_axis:]
        )
    else:
        slices_with_channel = slices
    
    return slices_with_channel