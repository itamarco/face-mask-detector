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
  grayMat = new cv.Mat(videoHeight, videoWidth, cv.CV_8UC1);

  faceClassifier = new cv.CascadeClassifier();
  faceClassifier.load('haarcascade_frontalface_default.xml');


  requestAnimationFrame(processVideo);
}

function processVideo() {
  let cap = new cv.VideoCapture(video);
  cap.read(srcMat);

  cv.flip(srcMat, srcMat, 1)
  cv.cvtColor(srcMat, grayMat, cv.COLOR_RGBA2GRAY);
  
  let faces = [];

  let size;

  let faceVect = new cv.RectVector();
  let faceMat = new cv.Mat();

  cv.pyrDown(grayMat, faceMat);
  cv.pyrDown(faceMat, faceMat);
  size = faceMat.size();

  faceClassifier.detectMultiScale(faceMat, faceVect);
  for (let i = 0; i < faceVect.size(); i++) {
    let face = faceVect.get(i);
    faces.push(new cv.Rect(face.x, face.y, face.width, face.height));
  }
  faceMat.delete();
  faceVect.delete();

  //canvasOutputCtx.drawImage(canvasInput, 0, 0, videoWidth, videoHeight);
  cv.imshow('canvasOutput', srcMat);
  drawResults(canvasOutputCtx, faces, 'blue', size);
  requestAnimationFrame(processVideo);
}

function drawResults(ctx, results, color, size) {
  for (let i = 0; i < results.length; ++i) {
    let rect = results[i];
    let xRatio = videoWidth / size.width;
    let yRatio = videoHeight / size.height;
    ctx.lineWidth = 3;
    ctx.strokeStyle = color;
    ctx.font = '48px serif';
    ctx.fillStyle = color;
    ctx.fillText('Text!', rect.x * xRatio, rect.y * yRatio - 4)
    ctx.strokeRect(rect.x * xRatio, rect.y * yRatio, rect.width * xRatio, rect.height * yRatio);
  }
}


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