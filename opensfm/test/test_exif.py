
import io
import pytest
import exifread
import datetime
import numpy as np
from opensfm import exif


@pytest.fixture
def dji_xmp_data_gimbal():
    return """
    <x:xmpmeta xmlns:x="adobe:ns:meta/">
        <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
            <rdf:Description rdf:about="" xmlns:drone-dji="http://www.dji.com/drone-dji/1.0/"
                drone-dji:GimbalYawDegree="+10.0"
                drone-dji:GimbalPitchDegree="-20.0"
                drone-dji:GimbalRollDegree="+30.0"
                drone-dji:Longitude="+5.0"
                drone-dji:Latitude="-6.0"
                drone-dji:AbsoluteAltitude="+100.0">
            </rdf:Description>
        </rdf:RDF>
    </x:xmpmeta>
    """


@pytest.fixture
def dji_xmp_data_flight():
    return """
    <x:xmpmeta xmlns:x="adobe:ns:meta/">
        <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
            <rdf:Description rdf:about="" xmlns:drone-dji="http://www.dji.com/drone-dji/1.0/"
                drone-dji:FlightYawDegree="+15.0"
                drone-dji:FlightPitchDegree="-25.0"
                drone-dji:FlightRollDegree="+35.0"
                drone-dji:Longitude="-5.0"
                drone-dji:Latitude="+6.0"
                drone-dji:AbsoluteAltitude="50.0">
            </rdf:Description>
        </rdf:RDF>
    </x:xmpmeta>
    """


@pytest.fixture
def dji_xmp_data_camera():
    return """
    <x:xmpmeta xmlns:x="adobe:ns:meta/">
        <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
            <rdf:Description rdf:about="" xmlns:Camera="http://pix4d.com/camera/1.0/"
                Camera:Yaw="12.0"
                Camera:Pitch="22.0"
                Camera:Roll="32.0"
                drone-dji:Longitude="1.0"
                drone-dji:Latitude="2.0"
                drone-dji:AbsoluteAltitude="3.0">
            </rdf:Description>
        </rdf:RDF>
    </x:xmpmeta>
    """


def create_exif_instance(xmp_content, monkeypatch):
    content = xmp_content.encode('utf-8')
    fileobj = io.BytesIO(content)
    fileobj.name = "test.jpg"

    monkeypatch.setattr("exifread.process_file", lambda f, details=False: {})

    return exif.EXIF(fileobj, lambda: (100, 100), "perspective")


def test_dji_parsing_latlon(dji_xmp_data_gimbal, monkeypatch):
    e = create_exif_instance(dji_xmp_data_gimbal, monkeypatch)
    assert e.has_dji_latlon()
    lon, lat = e.extract_dji_lon_lat()
    assert lon == 5.0
    assert lat == -6.0


def test_dji_parsing_altitude(dji_xmp_data_gimbal, monkeypatch):
    e = create_exif_instance(dji_xmp_data_gimbal, monkeypatch)
    assert e.has_dji_altitude()
    alt = e.extract_dji_altitude()
    assert alt == 100.0


def test_dji_parsing_public_geo(dji_xmp_data_gimbal, monkeypatch):
    e = create_exif_instance(dji_xmp_data_gimbal, monkeypatch)
    lon, lat = e.extract_lon_lat()
    assert lon == 5.0
    assert lat == -6.0


def test_dji_parsing_public_alt(dji_xmp_data_gimbal, monkeypatch):
    e = create_exif_instance(dji_xmp_data_gimbal, monkeypatch)
    alt = e.extract_altitude()
    assert alt == 100.0


def test_dji_parsing_opk(dji_xmp_data_gimbal, monkeypatch):
    e = create_exif_instance(dji_xmp_data_gimbal, monkeypatch)
    lon, lat = e.extract_lon_lat()
    alt = e.extract_altitude()
    geo_dict = {"latitude": lat, "longitude": lon, "altitude": alt}

    opk = e.extract_opk(geo_dict, e.extract_make(), e.extract_model())
    assert opk is not None


def test_dji_parsing_flight(dji_xmp_data_flight, monkeypatch):
    e = create_exif_instance(dji_xmp_data_flight, monkeypatch)

    lon, lat = e.extract_dji_lon_lat()
    assert lon == -5.0
    assert lat == 6.0

    alt = e.extract_dji_altitude()
    assert alt == 50.0

    geo_dict = {"latitude": lat, "longitude": lon, "altitude": alt}

    opk = e.extract_opk(geo_dict, e.extract_make(), e.extract_model())
    assert opk is not None


def test_dji_parsing_camera(dji_xmp_data_camera, monkeypatch):
    e = create_exif_instance(dji_xmp_data_camera, monkeypatch)

    geo_dict = {"latitude": 2.0, "longitude": 1.0, "altitude": 3.0}

    opk = e.extract_opk(geo_dict, e.extract_make(), e.extract_model())
    assert opk is not None


def test_dji_parsing_none(monkeypatch):
    # No XMP data
    content = b"dummy"
    fileobj = io.BytesIO(content)
    fileobj.name = "test.jpg"
    monkeypatch.setattr("exifread.process_file", lambda f, details=False: {})

    e = exif.EXIF(fileobj, lambda: (100, 100), "perspective")

    assert not e.has_dji_latlon()
    assert not e.has_dji_altitude()

    geo_dict = {"latitude": 0, "longitude": 0, "altitude": 0}
    opk = e.extract_opk(geo_dict, e.extract_make(), e.extract_model())
    assert opk is None


class MockTag:
    def __init__(self, values):
        self.values = values


def create_exif_with_tags(tags, monkeypatch):
    monkeypatch.setattr("exifread.process_file", lambda f, details=False: tags)
    fileobj = io.BytesIO(b"")
    fileobj.name = "test.jpg"

    monkeypatch.setattr("opensfm.exif.get_xmp", lambda f: [])
    return exif.EXIF(fileobj, lambda: (1000, 2000), "perspective")


def test_gps_parsing_standard(monkeypatch):
    tags = {
        "GPS GPSLatitude": MockTag([exifread.utils.Ratio(45, 1), exifread.utils.Ratio(30, 1), exifread.utils.Ratio(0, 1)]),
        "GPS GPSLatitudeRef": MockTag("N"),
        "GPS GPSLongitude": MockTag([exifread.utils.Ratio(10, 1), exifread.utils.Ratio(0, 1), exifread.utils.Ratio(0, 1)]),
        "GPS GPSLongitudeRef": MockTag("W"),
        "GPS GPSAltitude": MockTag([exifread.utils.Ratio(500, 1)]),
        "GPS GPSAltitudeRef": MockTag([1]),
    }

    e = create_exif_with_tags(tags, monkeypatch)

    lon, lat = e.extract_lon_lat()
    assert lat == 45.5
    assert lon == -10.0

    alt = e.extract_altitude()
    assert alt == -500.0


def test_focal_length_parsing(monkeypatch):
    tags = {
        "EXIF FocalLength": MockTag([exifread.utils.Ratio(24, 1)]),
        "EXIF FocalLengthIn35mmFilm": MockTag([exifread.utils.Ratio(35, 1)]),
        "EXIF ExifImageWidth": MockTag([4000]),
        "EXIF ExifImageLength": MockTag([3000]),
    }

    e = create_exif_with_tags(tags, monkeypatch)

    focal_35, focal_ratio = e.extract_focal()
    assert focal_35 == 35.0

    # Image is 4:3 (4000x3000), so film width is assumed 34mm
    assert abs(focal_ratio - (35.0 / 36.0)) < 0.01


def test_sensor_width_calculation(monkeypatch):
    tags = {
        # Focal 50mm
        "EXIF FocalLength": MockTag([exifread.utils.Ratio(50, 1)]),
        # 100 pixels per unit
        "EXIF FocalPlaneXResolution": MockTag([exifread.utils.Ratio(100, 1)]),
        "EXIF FocalPlaneResolutionUnit": MockTag([2]),  # Inches
        "EXIF ExifImageWidth": MockTag([1000]),  # 10 inches wide
        "EXIF ExifImageLength": MockTag([1000]),
    }

    # Width in pixels = 1000
    # Pixels per inch = 100
    # Sensor width in inches = 10
    # Sensor width in mm = 10 * 25.4 = 254.0

    e = create_exif_with_tags(tags, monkeypatch)
    sensor_width = e.extract_sensor_width()

    assert abs(sensor_width - 254.0) < 0.01

    _, focal_ratio = e.extract_focal()

    # Sensor width 254mm
    # Ratio = 50 / 254
    assert abs(focal_ratio - (50.0 / 254.0)) < 0.001


def test_orientation_parsing(monkeypatch):
    tags = {"Image Orientation": MockTag([6])}
    e = create_exif_with_tags(tags, monkeypatch)
    assert e.extract_orientation() == 6

    tags = {}
    e = create_exif_with_tags(tags, monkeypatch)
    assert e.extract_orientation() == 1


def test_make_model_parsing_image_tags(monkeypatch):
    tags = {
        "Image Make": MockTag("TestMake"),
        "Image Model": MockTag("TestModel"),
    }
    e = create_exif_with_tags(tags, monkeypatch)
    assert e.extract_make() == "TestMake"
    assert e.extract_model() == "TestModel"


def test_make_model_parsing_lens_tags(monkeypatch):
    tags = {
        "EXIF LensMake": MockTag("LensMake"),
        "EXIF LensModel": MockTag("LensModel"),
    }
    e = create_exif_with_tags(tags, monkeypatch)
    assert e.extract_make() == "LensMake"
    assert e.extract_model() == "LensModel"


def test_capture_time_gps(monkeypatch):
    tags = {
        "GPS GPSDate": MockTag("2021:01:01"),
        "GPS GPSTimeStamp": MockTag([exifread.utils.Ratio(12, 1), exifread.utils.Ratio(0, 1), exifread.utils.Ratio(0, 1)]),
    }
    e = create_exif_with_tags(tags, monkeypatch)

    delta = datetime.datetime(2021, 1, 1, 12, 0, 0) - \
        datetime.datetime(1970, 1, 1)
    expected = delta.total_seconds()
    assert e.extract_capture_time() == expected


def test_capture_time_exif(monkeypatch):
    tags = {
        "EXIF DateTimeOriginal": MockTag("2022:02:02 10:00:00"),
    }
    e = create_exif_with_tags(tags, monkeypatch)

    delta = datetime.datetime(2022, 2, 2, 10, 0, 0) - \
        datetime.datetime(1970, 1, 1)
    expected = delta.total_seconds()
    assert e.extract_capture_time() == expected


def test_extract_geo_structure_coord(monkeypatch):
    tags = {
        "GPS GPSLatitude": MockTag([exifread.utils.Ratio(10, 1), exifread.utils.Ratio(0, 1), exifread.utils.Ratio(0, 1)]),
        "GPS GPSLatitudeRef": MockTag("N"),
        "GPS GPSLongitude": MockTag([exifread.utils.Ratio(20, 1), exifread.utils.Ratio(0, 1), exifread.utils.Ratio(0, 1)]),
        "GPS GPSLongitudeRef": MockTag("E"),
    }

    e = create_exif_with_tags(tags, monkeypatch)
    geo = e.extract_geo()

    assert geo["latitude"] == 10.0
    assert geo["longitude"] == 20.0


def test_extract_geo_structure_alt_dop(monkeypatch):
    tags = {
        "GPS GPSAltitude": MockTag([exifread.utils.Ratio(100, 1)]),
        "GPS GPSDOP": MockTag([exifread.utils.Ratio(5, 10)]),  # 0.5
    }

    e = create_exif_with_tags(tags, monkeypatch)
    geo = e.extract_geo()

    assert geo["altitude"] == 100.0
    assert geo["dop"] == 0.5


def test_integration_extract_exif_from_file(monkeypatch):
    # This tests the module level function that wraps EXIF class
    tags = {
        "Image Make": MockTag("TestMake"),
        "Image Model": MockTag("TestModel"),
        "EXIF ExifImageWidth": MockTag([100]),
        "EXIF ExifImageLength": MockTag([100]),
    }

    monkeypatch.setattr("exifread.process_file", lambda f, details=False: tags)
    monkeypatch.setattr("opensfm.exif.get_xmp", lambda f: [])

    fileobj = io.BytesIO(b"")
    fileobj.name = "test.jpg"

    d = exif.extract_exif_from_file(
        fileobj, lambda: (100, 100), True, "perpspective")

    assert d["make"] == "TestMake"
    assert d["width"] == 100
    assert "camera" in d
