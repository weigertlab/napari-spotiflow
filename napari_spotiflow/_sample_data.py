import warnings

def _test_image_hybiss_2d():
    from spotiflow import sample_data
    return [(sample_data.test_image_hybiss_2d(), {"name": "hybiss_2d"})]

def _test_image_terra_2d():
    from spotiflow import sample_data
    return [(sample_data.test_image_terra_2d(), {"name": "terra_2d"})]

def _test_image_synth_3d():
    from spotiflow import sample_data
    return [(sample_data.test_image_synth_3d(), {"name": "synth_3d"})]

def _test_timelapse_telomeres_2d():
    from spotiflow import sample_data
    try:
        return [(sample_data.test_timelapse_telomeres_2d(), {"name": "telomeres_2d"})]
    except Exception as _:
        warnings.warn("Failed to load the Telomeres 2D+t dataset. Is the last version of Spotiflow installed?")
        return []
