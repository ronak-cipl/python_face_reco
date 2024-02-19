import face_recognition
import asyncio
import websockets
import json
import io

picture_of_ronak = face_recognition.load_image_file("ronak.jpeg")
ronak_face_encoding = face_recognition.face_encodings(picture_of_ronak)[0]

picture_of_pooja = face_recognition.load_image_file("pooja.jpeg")
pooja_face_encoding = face_recognition.face_encodings(picture_of_pooja)[0]

picture_of_mayur = face_recognition.load_image_file("mayur.jpeg")
mayur_face_encoding = face_recognition.face_encodings(picture_of_mayur)[0]

array_of_faces = [ronak_face_encoding, pooja_face_encoding, mayur_face_encoding]


async def websocket_handler(websocket):
    try:
        async for message in websocket:
            response = recognize_face(message)
            await websocket.send(json.dumps(response))
    except Exception as e:
        print(f"WebSocket Error: {str(e)}")


def recognize_face(message):
    try:
        unknown_picture = face_recognition.load_image_file(io.BytesIO(message))
        unknown_face_encodings = face_recognition.face_encodings(unknown_picture)
        if len(unknown_face_encodings) > 0:
            unknown_face_encoding = unknown_face_encodings[0]
        else:
            return {"status": True, "message": "No Face Detected", "data": 0}
        results = face_recognition.compare_faces(array_of_faces, unknown_face_encoding, 0.4)
        success = False
        for result in results:
            if result:
                success = True

        if success:
            index = results.index(True)
            name = ""
            if index == 0:
                name = "Ronak"
            elif index == 1:
                name = "Pooja"
            else:
                name = "Mayur"
            return {"status": True, "message": "Recognition successful", "name": name, "data": 2}
        else:
            return {"status": True, "message": "Recognition unsuccessful", "data": 1}

    except Exception as e:
        return {"status": False, "message": str(e)}


if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.run_until_complete(
        websockets.serve(websocket_handler, "192.168.0.102", 8765)
    )
    loop.run_forever()
