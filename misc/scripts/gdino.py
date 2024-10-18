from autodistill_grounded_sam import GroundedSAM
from autodistill.detection import CaptionOntology
import cv2
base_model = GroundedSAM(ontology=CaptionOntology({"squash_racket": "a racket players use to hit the squash ball, similar to a tennise racket but slightly less wide and slightly taller.", "squash_ball": "a very small black or white rubber ball that players hit and run to."}))
mask_annotator = sv.MaskAnnotator()

image = cv2.imread(image_name)

classes = base_model.ontology.classes()

detections = base_model.predict(image_name)

labels = [
    f"{classes[class_id]} {confidence:0.2f}"
    for _, _, confidence, class_id, _
    in detections
]

annotated_frame = mask_annotator.annotate(
    scene=image.copy(),
    detections=detections
)

sv.plot_image(annotated_frame, size=(8, 8))