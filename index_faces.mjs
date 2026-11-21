import fs from 'fs';
import '@tensorflow/tfjs-backend-wasm';
import * as faceapi from '@vladmandic/face-api/dist/face-api.node-wasm.js';
import canvas from 'canvas';

const { Canvas, Image, ImageData } = canvas;
faceapi.env.monkeyPatch({ Canvas, Image, ImageData });

const FOLDER_ID = "PASTE_MOHAMMED_NOOH_DRIVE_FOLDER_ID";
const API_KEY = "AIzaSyD4eAgS3-sVEfKeoW21lXrqw6ENbOPP9qY";
const MODEL_URL = "https://cdn.jsdelivr.net/npm/@vladmandic/face-api/model";

async function fetchFiles(query) {
  let files = [];
  let pageToken = "";
  while (true) {
    const url = `https://www.googleapis.com/drive/v3/files?q=${encodeURIComponent(query)}&fields=nextPageToken,files(id,name)&pageSize=1000&key=${API_KEY}${pageToken ? `&pageToken=${pageToken}` : ""}`;
    const res = await fetch(url).then(r => r.json());
    if (res.error) throw new Error(res.error.message);
    files.push(...(res.files || []));
    if (!res.nextPageToken) break;
    pageToken = res.nextPageToken;
  }
  return files;
}

async function listAllDriveImages(rootFolderId) {
  const folderIds = [rootFolderId];
  const checkedFolders = new Set();
  const allImages = [];

  while (folderIds.length > 0) {
    const currentId = folderIds.shift();
    if (checkedFolders.has(currentId)) continue;
    checkedFolders.add(currentId);

    const subfolders = await fetchFiles(`'${currentId}' in parents and mimeType = 'application/vnd.google-apps.folder' and trashed = false`);
    for (const sub of subfolders) {
      folderIds.push(sub.id);
    }

    const images = await fetchFiles(`'${currentId}' in parents and mimeType contains 'image/' and trashed = false`);
    allImages.push(...images);
  }
  return allImages;
}

async function run() {
  console.log("Initializing WASM backend...");
  await faceapi.tf.setBackend('wasm');
  await faceapi.tf.ready();

  console.log("Loading AI models...");
  await Promise.all([
    faceapi.nets.ssdMobilenetv1.loadFromUri(MODEL_URL),
    faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL),
    faceapi.nets.faceRecognitionNet.loadFromUri(MODEL_URL)
  ]);

  console.log("Fetching photo list from Google Drive...");
  const images = await listAllDriveImages(FOLDER_ID);
  console.log(`Found ${images.length} photos. Scanning faces...`);

  const photos = [];
  for (let i = 0; i < images.length; i++) {
    const imgData = images[i];
    try {
      const thumbUrl = `https://drive.google.com/thumbnail?id=${imgData.id}&sz=w800`;
      const img = await canvas.loadImage(thumbUrl);
      const detections = await faceapi.detectAllFaces(img).withFaceLandmarks().withFaceDescriptors();

      photos.push({
        id: imgData.id,
        name: imgData.name,
        faces: detections.map(d => Array.from(d.descriptor))
      });
      console.log(`[${i + 1}/${images.length}] ${imgData.name} -> ${detections.length} face(s)`);
    } catch (err) {
      photos.push({
        id: imgData.id,
        name: imgData.name,
        faces: []
      });
      console.log(`[${i + 1}/${images.length}] Included without face scan (${imgData.name}): ${err.message}`);
    }
  }

  fs.writeFileSync("faces.json", JSON.stringify({ folderId: FOLDER_ID, photos }));
  console.log(`Done! Saved ${photos.length} photos to faces.json`);
}

run().catch(err => {
  console.error("ERROR:", err);
  process.exit(1);
});
