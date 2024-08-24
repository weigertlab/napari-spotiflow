def _test_image_hybiss_2d():
    from spotiflow import sample_data
    return [(sample_data.test_image_hybiss_2d(), {"name": "hybiss_2d"})]

def _test_image_terra_2d():
    from spotiflow import sample_data
    return [(sample_data.test_image_terra_2d(), {"name": "terra_2d"})]

def _test_image_synth_3d():
    from spotiflow import sample_data
    return [(sample_data.test_image_synth_3d(), {"name": "synth_3d"})]
