import pytest
from starlette.websockets import WebSocketDisconnect

from tests.conftest import BLUE, DIM_BLUE, RED, frame_b64, jpeg, side_by_side, solid


def upload(name: str, data: bytes):
    return (name, data, "image/png")


def enroll(client, name: str, color, **kw):
    return client.post("/enroll", data={"name": name}, files={"image": upload("p.png", jpeg(solid(color)))}, **kw)


def test_compare_same_and_different(client):
    same = client.post("/compare", files={"image_a": upload("a.png", jpeg(solid(BLUE))),
                                          "image_b": upload("b.png", jpeg(solid(BLUE)))}).json()
    assert same["verdict"] == "same_person" and same["match_percent"] == 100.0
    diff = client.post("/compare", files={"image_a": upload("a.png", jpeg(solid(BLUE))),
                                          "image_b": upload("b.png", jpeg(solid(RED)))}).json()
    assert diff["verdict"] == "different_person"


def test_bad_inputs(client):
    no_face = client.post("/compare", files={"image_a": upload("a.png", jpeg(solid((0, 0, 0)))),
                                             "image_b": upload("b.png", jpeg(solid(BLUE)))})
    assert no_face.status_code == 422 and "image A" in no_face.json()["detail"]
    garbage = client.post("/identify", files={"image": upload("x.png", b"not an image")})
    assert garbage.status_code == 400
    empty = client.post("/identify", files={"image": upload("x.png", b"")})
    assert empty.status_code == 400
    assert client.post("/identify?top_k=500", files={"image": upload("x.png", jpeg(solid(BLUE)))}).status_code == 422


def test_upload_size_limit(make_client):
    client = make_client(max_image_bytes=1000)
    big = jpeg(solid(BLUE, width=64, height=64)) + b"\0" * 2000
    assert client.post("/identify", files={"image": upload("big.png", big)}).status_code == 413


def test_enroll_validation(client):
    assert enroll(client, "   ", BLUE).status_code == 422
    assert enroll(client, "x" * 101, BLUE).status_code == 422
    assert enroll(client, "Asha", DIM_BLUE).status_code == 422  # detector not confident enough
    ok = enroll(client, "  Asha   Rao ", BLUE).json()
    assert ok["name"] == "Asha Rao"


def test_enroll_identify_list_delete(client):
    asha = enroll(client, "Asha", BLUE).json()["identity_id"]
    group = client.post("/identify", files={"image": upload("g.png", side_by_side(BLUE, RED))}).json()["faces"]
    assert group[0]["identity"]["name"] == "Asha" and group[0]["identity"]["confident"]
    assert group[1]["identity"] is None  # unknown person
    assert client.get("/people").json() == {"people": {"Asha": [asha]}}
    assert client.get("/health").json()["enrolled"] == 1
    assert client.delete(f"/people/{asha}").json() == {"deleted": asha}
    assert client.delete(f"/people/{asha}").status_code == 404


def test_api_key_is_enforced(make_client):
    client = make_client(api_key="s3cret-test-key")
    assert client.get("/people").status_code == 401
    assert client.get("/people", headers={"X-API-Key": "wrong"}).status_code == 401
    assert enroll(client, "Asha", BLUE).status_code == 401
    assert client.get("/people", headers={"X-API-Key": "s3cret-test-key"}).status_code == 200
    health = client.get("/health").json()  # stays open for container health checks
    assert health["status"] == "ok" and health["auth_required"] is True


def test_no_cors_unless_configured(make_client):
    preflight = {"Origin": "https://evil.example", "Access-Control-Request-Method": "POST"}
    assert "access-control-allow-origin" not in make_client().options("/enroll", headers=preflight).headers
    allowed = make_client(cors_origins=("https://app.example",))
    ok = allowed.options("/enroll", headers={**preflight, "Origin": "https://app.example"})
    assert ok.headers["access-control-allow-origin"] == "https://app.example"


def test_websocket_identifies_frames(client):
    enroll(client, "Asha", BLUE)
    with client.websocket_connect("/ws/identify") as ws:
        ws.send_text(frame_b64(solid(BLUE)))
        assert ws.receive_json()["faces"][0]["name"] == "Asha"
        ws.send_text("%%%not-base64%%%")
        assert ws.receive_json() == {"error": "bad_frame"}


def test_websocket_rejects_other_sites(client):
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/ws/identify", headers={"Origin": "https://evil.example"}) as ws:
            ws.receive_json()
    with client.websocket_connect("/ws/identify", headers={"Origin": "http://testserver"}) as ws:
        ws.send_text(frame_b64(solid(RED)))
        assert ws.receive_json()["faces"][0]["name"] is None


def test_websocket_needs_key_first(make_client):
    client = make_client(api_key="s3cret-test-key")
    with client.websocket_connect("/ws/identify") as ws:
        ws.send_text(frame_b64(solid(BLUE)))  # a frame instead of the key
        assert ws.receive_json() == {"error": "unauthorized"}
    with client.websocket_connect("/ws/identify") as ws:
        ws.send_text("s3cret-test-key")
        assert ws.receive_json() == {"ok": "authorized"}
        ws.send_text(frame_b64(solid(BLUE)))
        assert ws.receive_json()["faces"][0]["name"] is None


def test_web_ui_is_served(client):
    page = client.get("/")
    assert page.status_code == 200 and "Face Match" in page.text
    assert ".innerHTML =" not in page.text  # enrolled names are user input: never rendered as HTML
