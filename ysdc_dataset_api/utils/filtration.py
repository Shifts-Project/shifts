def request_is_valid(scene, request):
    # Checks whether all request.track_id is present at all future tracks steps
    for i in range(len(scene.future_vehicle_tracks)):
        current_track_ids = {
            t.track_id
            for t in scene.future_vehicle_tracks[i].tracks
        }
        if request.track_id not in current_track_ids:
            return False
    return True
