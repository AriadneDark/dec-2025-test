def propagate_in_video(predictor, session_id, frame_idx, prop_direction="both"):
    """
    Propagates from frame 'frame_idx' to all video
    """
    outputs_per_frame = {}
    for response in predictor.handle_stream_request(
        request=dict(
            type="propagate_in_video",
            propagation_direction=prop_direction,
            start_frame_idx=frame_idx,
            session_id=session_id,
        )
    ):
        outputs_per_frame[response["frame_index"]] = response["outputs"]

    return outputs_per_frame


def segment_on_vigeo(video_predictor, image_path, prompt, prompt_idx=0):
    """
    Run Sam3 inference on image sequence or video
    Params:
        video_predictor - Sam3 video predictor
        image_path - path to a JPEG folder or an MP4 video file
        prompt - dict with text or bbox prompt
        prompt_idx - frame for which prompt will be used
    Returns: predicted output

    """
    video_path = image_path  # a JPEG folder or an MP4 video file
    # Start a session
    response = video_predictor.handle_request(
        request=dict(
            type="start_session",
            resource_path=video_path,
        )
    )
    session_id = response["session_id"]
    # add text prompt
    if "text" in prompt.keys():
        response = video_predictor.handle_request(
            request=dict(
                type="add_prompt",
                session_id=session_id,
                frame_index=prompt_idx,  # Arbitrary frame index
                text=prompt["text"],
            )
        )
    # add bbox prompt
    if "bounding_boxes" in prompt.keys():
        response = video_predictor.handle_request(
            request=dict(
                type="add_prompt",
                session_id=session_id,
                frame_index=prompt_idx,  # Arbitrary frame index
                bounding_boxes=prompt["bounding_boxes"],
                bounding_box_labels=prompt["box_labels"],
            )
        )
    # propagate on video
    outputs_per_frame = propagate_in_video(video_predictor, session_id, prompt_idx)
    # close session
    response = video_predictor.handle_request(
        request=dict(
            type="close_session",
            session_id=session_id,
        )
    )
    return outputs_per_frame
