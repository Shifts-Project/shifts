class RendererBase:
    def __init__(self, spec):
        self._spec = spec

    def render(self, track_id, scene):
        raise NotImplementedError()


class DummyRenderer(RendererBase):
    def render(self, track_id, scene):
        return track_id, scene
