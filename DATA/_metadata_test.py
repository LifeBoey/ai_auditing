
from PIL import Image, ImageStat
from PIL.ExifTags import TAGS
from pathlib import Path
import json

IMG_DIR = Path("data_sub500_v2")
NEW_IMG_DIR = Path("data_sub500_v2_metadata")


# Formats to process
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".tif", ".tiff", ".webp"}


def get_image_metadata(image_path):
    """Extract basic image properties and EXIF metadata."""
    with Image.open(image_path) as image:
        metadata = {
            "Filename": image.filename,
            "Format": image.format,
            "Mode": image.mode,
            "Size": image.size,
        }

        exif_data = image.getexif()

        for tag_id, value in exif_data.items():
            tag_name = TAGS.get(tag_id, tag_id)

            if isinstance(value, bytes):
                value = value.decode(errors="replace")

            metadata[tag_name] = value

        return metadata


def calculate_brightness(image):
    """Calculate mean grayscale brightness on a 0-255 scale."""
    grayscale = image.convert("L")
    return ImageStat.Stat(grayscale).mean[0]

# Mean grayscale brightness threshold (0 = black, 255 = white)
BRIGHTNESS_THRESHOLD1 = 90
BRIGHTNESS_THRESHOLD2 = 130
def brightness_check(brightness_value):
    if brightness_value <= BRIGHTNESS_THRESHOLD1:
        return "dark"
    elif brightness_value >= BRIGHTNESS_THRESHOLD2: 
        return "bright"
    else:
        return "normal"

def process_image(image_path, output_dir):
    """Calculate brightness, store it in EXIF UserComment, and save."""
    with Image.open(image_path) as image:
        # Calculate brightness before modifying the image
        brightness_value = calculate_brightness(image)

        brightness_label = brightness_check(brightness_value)

        # Custom metadata to embed
        custom_metadata = {
            "brightness": brightness_label,
            "brightness_value": round(brightness_value, 2),
        }

        # Convert metadata dictionary to JSON
        json_string = json.dumps(custom_metadata)

        # Read existing EXIF metadata
        exif_data = image.getexif()

        # EXIF tag 37510 = UserComment.
        # The 8-byte ASCII prefix identifies the comment encoding.
        exif_data[37510] = (
            b"ASCII\x00\x00\x00" + json_string.encode("ascii")
        )

        # Save under the same filename in the output directory
        output_path = output_dir / image_path.name

        # Preserve the original image format and attach EXIF
        image.save(
            output_path,
            format=image.format,
            exif=exif_data,
        )

        print(
            f"Processed: {image_path.name} | "
            f"Brightness: {brightness_value:.2f} "
            f"({brightness_label})"
        )


def main():
    # Validate input directory
    if not IMG_DIR.is_dir():
        raise FileNotFoundError(
            f"Input directory does not exist: {IMG_DIR}"
        )

    # Create output directory if needed
    NEW_IMG_DIR.mkdir(parents=True, exist_ok=True)

    # Process images
    for image_path in IMG_DIR.iterdir():
        if not image_path.is_file():
            continue

        if image_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        try:
            process_image(image_path, NEW_IMG_DIR)
        except Exception as error:
            print(f"Failed to process {image_path.name}: {error}")


if __name__ == "__main__":
    main()

# photo_details = get_image_metadata("metadata_000000001296.jpg")
# for key, val in photo_details.items():
#     print(f"{key}: {val}")


# if exif_raw:
#     for tag_id, value in exif_raw.items():
#         tag_name = TAGS.get(tag_id, tag_id)
        
#         # If the tag contains GPS information, parse it separately
#         if tag_name == "GPSInfo":
#             gps_data = {}
#             for gps_id in value:
#                 gps_tag = GPSTAGS.get(gps_id, gps_id)
#                 gps_data[gps_tag] = value[gps_id]
#             print("GPS Metadata:", gps_data)
#         else:
#             print(f"{tag_name}: {value}")
