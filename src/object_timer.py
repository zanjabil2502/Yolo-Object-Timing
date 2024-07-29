from collections import defaultdict

import cv2

from ultralytics.utils.checks import check_imshow, check_requirements
from ultralytics.utils.plotting import Annotator, colors

check_requirements("shapely>=2.0.0")

from shapely.geometry import Polygon, Point


class ObjectTimer:
    """A class to manage the timing of objects in a real-time video stream based on their tracks."""

    def __init__(
        self,
        names,
        reg_pts=None,
        fps=None,
        time_reg_color=(255, 0, 255),
        time_txt_color=(0, 0, 0),
        time_bg_color=(255, 255, 255),
        line_thickness=2,
        track_thickness=2,
        view_img=False,
        view_single_time=True,
        view_average_time=True,
        view_max_time=True,
        draw_tracks=False,
        track_color=None,
        region_thickness=5,
        line_dist_thresh=15,
        cls_txtdisplay_gap=50,
    ):
        """
        Initializes the ObjectCounter with various tracking and counting parameters.

        Args:
            names (dict): Dictionary of class names.
            reg_pts (list): List of points defining the counting region.
            fps (int): FPS input video source.
            time_reg_color (tuple): RGB color of the timing region.
            time_txt_color (tuple): RGB color of the time text.
            time_bg_color (tuple): RGB color of the time text background.
            line_thickness (int): Line thickness for bounding boxes.
            track_thickness (int): Thickness of the track lines.
            view_img (bool): Flag to control whether to display the video stream.
            view_single_time (bool): Flag to control whether to display the in counts on the video stream.
            view_average_time (bool): Flag to control whether to display the out counts on the video stream.
            draw_tracks (bool): Flag to control whether to draw the object tracks.
            track_color (tuple): RGB color of the tracks.
            region_thickness (int): Thickness of the object counting region.
            line_dist_thresh (int): Euclidean distance threshold for line counter.
            cls_txtdisplay_gap (int): Display gap between each class count.
        """

        # Mouse events
        self.is_drawing = False
        self.selected_point = None

        # Region & Line Information
        if fps is None:
            raise ValueError("FPS value must be filled in.")

        self.fps = fps
        self.reg_pts = [(20, 400), (1260, 400)] if reg_pts is None else reg_pts
        self.line_dist_thresh = line_dist_thresh
        self.counting_region = None
        self.region_color = time_reg_color
        self.region_thickness = region_thickness

        # Image and annotation Information
        self.im0 = None
        self.tf = line_thickness
        self.view_img = view_img
        self.view_single_time = view_single_time
        self.view_average_time = view_average_time
        self.view_max_time = view_max_time

        self.names = names  # Classes names
        self.annotator = None  # Annotator
        self.window_name = "Object Timing by Zanjabila"

        # Object counting Information
        self.in_counts = 0
        self.out_counts = 0
        self.count_ids = []
        self.object_first_frame = {}
        self.object_timing_data = {}
        self.count_txt_thickness = 0
        self.time_txt_color = time_txt_color
        self.time_bg_color = time_bg_color
        self.cls_txtdisplay_gap = cls_txtdisplay_gap
        self.fontsize = 0.6

        # Tracks info
        self.track_history = defaultdict(list)
        self.track_thickness = track_thickness
        self.draw_tracks = draw_tracks
        self.track_color = track_color

        # Check if environment supports imshow
        self.env_check = check_imshow(warn=True)

        # Initialize counting region
        if len(self.reg_pts) < 3:
            raise ValueError(
                "Region points must be more than 3 points to be a polygon."
            )

        print("Polygon Counter Initiated.")
        self.counting_region = Polygon(self.reg_pts)

    def mouse_event_for_region(self, event, x, y, flags, params):
        """
        Handles mouse events for defining and moving the counting region in a real-time video stream.

        Args:
            event (int): The type of mouse event (e.g., cv2.EVENT_MOUSEMOVE, cv2.EVENT_LBUTTONDOWN, etc.).
            x (int): The x-coordinate of the mouse pointer.
            y (int): The y-coordinate of the mouse pointer.
            flags (int): Any associated event flags (e.g., cv2.EVENT_FLAG_CTRLKEY,  cv2.EVENT_FLAG_SHIFTKEY, etc.).
            params (dict): Additional parameters for the function.
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            for i, point in enumerate(self.reg_pts):
                if (
                    isinstance(point, (tuple, list))
                    and len(point) >= 2
                    and (abs(x - point[0]) < 10 and abs(y - point[1]) < 10)
                ):
                    self.selected_point = i
                    self.is_drawing = True
                    break

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.is_drawing and self.selected_point is not None:
                self.reg_pts[self.selected_point] = (x, y)
                self.counting_region = Polygon(self.reg_pts)

        elif event == cv2.EVENT_LBUTTONUP:
            self.is_drawing = False
            self.selected_point = None

    def extract_and_process_tracks(self, tracks, num_frame):
        """Extracts and processes tracks for object counting in a video stream."""

        # Annotator Init and region drawing
        self.annotator = Annotator(self.im0, self.tf, self.names)

        # Draw region or line
        self.annotator.draw_region(
            reg_pts=self.reg_pts,
            color=self.region_color,
            thickness=self.region_thickness,
        )

        if tracks[0].boxes.id is not None:
            boxes = tracks[0].boxes.xyxy.cpu()
            clss = tracks[0].boxes.cls.cpu().tolist()
            track_ids = tracks[0].boxes.id.int().cpu().tolist()

            # Extract tracks
            for box, track_id, cls in zip(boxes, track_ids, clss):

                # Draw Tracks
                centroid_object = (
                    float((box[0] + box[2]) / 2),
                    float((box[1] + box[3]) / 2),
                )

                track_line = self.track_history[track_id]
                track_line.append(centroid_object)
                if len(track_line) > 30:
                    track_line.pop(0)

                # Draw track trails
                if self.draw_tracks:
                    self.annotator.draw_centroid_and_tracks(
                        track_line,
                        color=self.track_color or colors(int(track_id), True),
                        track_thickness=self.track_thickness,
                    )

                prev_position = (
                    self.track_history[track_id][-2]
                    if len(self.track_history[track_id]) > 1
                    else None
                )

                is_inside = self.counting_region.contains(Point(track_line[-1]))

                if is_inside:
                    # Store class info
                    label = f"{self.names[cls]}#{track_id}"
                    if label not in self.object_first_frame:
                        self.object_first_frame[label] = num_frame

                    if prev_position is not None:
                        diff = num_frame - self.object_first_frame[label]
                        time = int(diff * (1 / self.fps))

                        text_time = "Time: " + str(time) + " sec"
                        label_box = f"{label} | {text_time}"

                        # Draw bounding box
                        self.object_timing_data[label_box] = time
                        self.annotator.box_label(
                            box,
                            label=label_box,
                            color=colors(int(track_id), True),
                        )

        labels_dict = {}

        if len(self.object_timing_data) > 0 and self.view_average_time:
            values = self.object_timing_data.values()
            average = round(sum(values) / len(values), 2)
            labels_dict["Average Time"] = str(average) + " Sec"

        if len(self.object_timing_data) > 0 and self.view_max_time:
            values = self.object_timing_data.values()
            labels_dict["Max Time"] = str(max(values)) + " Sec"

        if labels_dict:
            self.annotator.display_analytics(
                self.im0, labels_dict, self.time_txt_color, self.time_bg_color, 10
            )

    def display_frames(self):
        """Displays the current frame with annotations and regions in a window."""
        if self.env_check:
            cv2.namedWindow(self.window_name)
            if len(self.reg_pts) == 4:  # only add mouse event If user drawn region
                cv2.setMouseCallback(
                    self.window_name,
                    self.mouse_event_for_region,
                    {"region_points": self.reg_pts},
                )
            cv2.imshow(self.window_name, self.im0)
            # Break Window
            if cv2.waitKey(1) & 0xFF == ord("q"):
                return

    def start_timing(self, im0, tracks, num_frame):
        """
        Main function to start the object counting process.

        Args:
            im0 (ndarray): Current frame from the video stream.
            tracks (list): List of tracks obtained from the object tracking process.
        """
        self.im0 = im0  # store image
        self.extract_and_process_tracks(
            tracks, num_frame
        )  # draw region even if no objects

        if self.view_img:
            self.display_frames()
        return self.im0


if __name__ == "__main__":
    classes_names = {0: "person", 1: "car"}  # example class names
    ObjectTimer(classes_names)
