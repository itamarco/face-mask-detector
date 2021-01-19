let videoWidth, videoHeight;

// whether streaming video from the camera.
let streaming = false;

let video = document.getElementById('video');
let canvasOutput = document.getElementById('canvasOutput');
let canvasOutputCtx = canvasOutput.getContext('2d');
let stream = null;

let detectFace = document.getElementById('face');
let detectEye = document.getElementById('eye');

function startCamera() {
  if (streaming) return;
  navigator.mediaDevices.getUserMedia({ video: true, audio: false })
    .then(function (s) {
      stream = s;
      video.srcObject = s;
      video.play();
    })
    .catch(function (err) {
      console.log("An error occured! " + err);
    });

  video.addEventListener("canplay", function (ev) {
    if (!streaming) {
      videoWidth = video.videoWidth;
      videoHeight = video.videoHeight;
      video.setAttribute("width", videoWidth);
      video.setAttribute("height", videoHeight);
      canvasOutput.width = videoWidth;
      canvasOutput.height = videoHeight;
      streaming = true;
    }
    startVideoProcessing();
  }, false);
}

let faceClassifier = null;
let eyeClassifier = null;

let canvasInput = null;
let canvasInputCtx = null;

let canvasBuffer = null;
let canvasBufferCtx = null;

function startVideoProcessing() {
  if (!streaming) { console.warn("Please startup your webcam"); return; }


  srcMat = new cv.Mat(videoHeight, videoWidth, cv.CV_8UC4);
  dstMat = new cv.Mat(videoHeight, videoWidth, cv.CV_8UC4);
  grayMat = new cv.Mat(videoHeight, videoWidth, cv.CV_8UC1);

  faceClassifier = new cv.CascadeClassifier();
  faceClassifier.load('haarcascade_frontalface_default.xml');


  requestAnimationFrame(processVideo);
}

function processVideo() {
  let cap = new cv.VideoCapture(video);
  cap.read(srcMat);

  cv.flip(srcMat, srcMat, 1)
  srcMat.copyTo(dstMat)


  cv.cvtColor(srcMat, grayMat, cv.COLOR_RGBA2GRAY);

  let faces = [];

  let size;

  let faceVect = new cv.RectVector();
  let faceMat = new cv.Mat();

  cv.pyrDown(grayMat, faceMat);
  cv.pyrDown(faceMat, faceMat);
  size = faceMat.size();

  faceClassifier.detectMultiScale(faceMat, faceVect);

  addFaces(dstMat, faceVect, size);

  faceMat.delete();
  faceVect.delete();





  cv.imshow('canvasOutput', dstMat);
  requestAnimationFrame(processVideo);
}

function addFaces(dstMat, faces, size) {

  let xRatio = videoWidth / size.width;
  let yRatio = videoHeight / size.height;

  for (let i = 0; i < faces.size(); ++i) {
    let face = faces.get(i);
    let point1 = new cv.Point(face.x * xRatio, face.y * yRatio);
    let point2 = new cv.Point(face.x * xRatio + face.width * xRatio, face.y * yRatio + face.height * yRatio);
    cv.rectangle(dstMat, point1, point2, [255, 0, 0, 255]);
    cv.putText(dstMat, 'ynkeee', point1, cv.FONT_HERSHEY_SIMPLEX, 0.8, [255, 255, 0, 255], 2)
  }
}

// function drawResults(dstMat, faces, color, size) {
//   for (let i = 0; i < results.length; ++i) {
//     let rect = results[i];
//     let xRatio = videoWidth / size.width;
//     let yRatio = videoHeight / size.height;
//     ctx.lineWidth = 3;
//     ctx.strokeStyle = color;
//     ctx.strokeRect(rect.x * xRatio, rect.y * yRatio, rect.width * xRatio, rect.height * yRatio);
//   }
// }


function stopCamera() {
  if (!streaming) return;
  document.getElementById("canvasOutput").getContext("2d").clearRect(0, 0, width, height);
  video.pause();
  video.srcObject = null;
  stream.getVideoTracks()[0].stop();
  streaming = false;
}


function opencvIsReady() {
  console.log('OpenCV.js is ready');
  startCamera();
}