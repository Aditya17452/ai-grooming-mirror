# import cv2
# from grooming.face import analyze_face
# from grooming.torso import extract_torso
# from grooming.tie import detect_tie
# from grooming.color import color_contrast
# from grooming.texture import texture_quality
# from grooming.scorer import grooming_score
# from ui.dashboard import render

# cap = cv2.VideoCapture(0)

# ret, frame = cap.read()

# face = analyze_face(frame)
# torso = extract_torso(frame)
# tie = detect_tie(frame) if torso is not None else False

# signals = {
#     "face": face["present"],
#     "upper_body": torso is not None,
#     "tie": tie,
#     "color": True,
#     "texture": True
# }

# score = grooming_score(signals)

# feedback = []
# if not tie:
#     feedback.append("Wearing a tie improves professional appearance in pharma interactions.")

# render(score, feedback)
