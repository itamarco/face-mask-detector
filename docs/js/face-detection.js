let videoWidth, videoHeight;

// whether streaming video from the camera.
let streaming = false;

let video = document.getElementById('video');
let canvasOutput = document.getElementById('canvasOutput');
let canvasOutputCtx = canvasOutput.getContext('2d');
let stream = null;

let detectFace = document.getElementById('face');
let detectEye = document.getElementById('eye');

let hasMaskByPredictionClass = {
  0: true,
  1: false
}



function startCamera() {
  if (streaming) return;
  navigator.mediaDevices.getUserMedia({ video: true, audio: false })
    .then(function (s) {
      stream = s;
      video.srcObject = s;
      video.play();
    })
    .catch(function (err) {
      printOutput("An error occured! " + err.msg);
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

  let faceVect = new cv.RectVector();
  let faceMat = new cv.Mat();

  cv.pyrDown(grayMat, faceMat);
  cv.pyrDown(faceMat, faceMat);
  let scale = faceMat.size();

  cv.imshow('canvasOutput', srcMat);
  faceClassifier.detectMultiScale(faceMat, faceVect);
  for (let i = 0; i < faceVect.size(); i++) {
    let face = faceVect.get(i);
    faces.push(new cv.Rect(face.x, face.y, face.width, face.height));
  }
  faceMat.delete();
  faceVect.delete();

  
  drawResults(canvasOutputCtx, faces, scale);
  requestAnimationFrame(processVideo);
}

function drawResults(ctx, results, scale) {
  for (let i = 0; i < results.length; ++i) {
    let faceRect = results[i];
    let xScale = videoWidth / scale.width;
    let yScale = videoHeight / scale.height;

    let [x,y,xWidth,yWidth] = [faceRect.x * xScale -3 , faceRect.y * yScale + 3 , faceRect.width * xScale + 3, faceRect.height * yScale];
    let color,text;

    let {classIndex, score} = detectMask(canvasOutputCtx,x,y,xWidth,yWidth);
    let hasMask = hasMaskByPredictionClass[classIndex];
    score = score.toFixed(3);

    if (hasMask) {
      color = 'rgb(0,225,0)';
      text = 'with mask ' + score;
    } else {
      color = 'red';
      text = 'no mask ' + score;
    }
    ctx.lineWidth = 4;
    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.fillRect(x,y-50,xWidth,50);
    ctx.strokeRect(x,y,xWidth,yWidth);
    ctx.font = 'bold 32px arial';
    ctx.fillStyle = 'white';
    ctx.fillText(text, x, y - 4)
  }
}

function detectMask(canvasCtx, x, y, xWidth, yWidth) {
    //get image data
    let faceImage = canvasOutputCtx.getImageData(x,y,xWidth,yWidth);
    let tensor = proccessImage(faceImage);
    let prediction = predict(tensor);
    console.log(prediction)
    return prediction;
}

function stopCamera() {
  if (!streaming) return;
  document.getElementById("canvasOutput").getContext("2d").clearRect(0, 0, canvasOutput.width, canvasOutput.height);
  video.pause();
  video.srcObject = null;
  stream.getVideoTracks()[0].stop();
  streaming = false;
  clearOutput();
  printOutput('Reload the page to start over')
}


function opencvIsReady() {
  printOutput('OpenCV is ready');
  printOutput('Please allow camera access');
  startCamera();
}