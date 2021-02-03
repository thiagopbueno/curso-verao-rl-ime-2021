class PiecewiseLinearSchedule:

    def __init__(self, *points):
        assert len(points) > 1, "schedule needs at least 2 points"

        steps = [p[0] for p in points]
        assert sorted(steps) == steps, "steps in schedule must be sorted in ascending order"

        self._points = points

    def __call__(self, timestep):
        initial_step, initial_value = self._points[0]
        if timestep < initial_step:
            return initial_value

        for (t1, v1), (t2, v2) in zip(self._points[:-1], self._points[1:]):
            if t1 <= timestep < t2:
                ratio = (timestep - t1) / (t2 - t1)
                return v1 + ratio * (v2 - v1)

        _, final_value = self._points[-1]
        return final_value