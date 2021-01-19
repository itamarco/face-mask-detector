
const TARGET_HEIGHT = 140
const TARGET_WIDTH = 140

let model;
function predict(imageTensor) {
    const prediction = model.predict(imageTensor);
    const classIndex = prediction.argMax(-1).dataSync()[0];
    const score = prediction.dataSync()[classIndex];
    return {
        classIndex,
        score
    };
}

function classifyImage(event) {
    event.preventDefault();
    const imgEl = document.getElementById('testImg');
    let tensor = proccessImage(imgEl);

    const { classIndex, score } = predict(tensor);
    console.log(classIndex, score);
}

function proccessImage(image) {
    let tensor = tf.browser.fromPixels(image)
        .resizeBilinear([TARGET_WIDTH, TARGET_HEIGHT])
        .toFloat()
        .div(255)
        .reshape([1, TARGET_WIDTH, TARGET_HEIGHT, 3]);

    return tensor;
}

function displayImage(obj) {
    if (FileReader) {
        var reader = new FileReader();
        reader.readAsDataURL(obj.files[0]);
        reader.onload = function (e) {
            var image = new Image();
            image.src = e.target.result;
            image.onload = function () {
                document.getElementById('testImg').src = image.src;
            };
        }
    } else {
        // Not supported
    }
}

async function liveDetect() {
    cosnt webcamConfig: webcamConfig = {

    }
    const webcamElement = document.getElementById('webcam');
    const webcam = await tf.data.webcam(webcamElement);
    while (true) {
        const img = await webcam.capture()
        .resizeWidth
        .toFloat()
        .div(255)
        .reshape([1, TARGET_WIDTH, TARGET_HEIGHT, 3]);
        //const tensor = proccessImage(img);
        const { classIndex, score } = predict(img);


        document.getElementById('outpur').innerText = `
            prediction: ${result[0].className}\n
            probability: ${result[0].probability}
            `;
        // Dispose the tensor to release the memory.
        img.dispose();
        //tensor.dispose();

        // Give some breathing room by waiting for the next animation frame to
        // fire.
        await tf.nextFrame();
    }
}

function printOutput(msg) {
    document.getElementById('output').innerHTML = msg;
}


async function app() {
    printOutput('Loading face-mask model..');


    // Load the model.
    model = await tf.loadLayersModel('https://raw.githubusercontent.com/itamarco/face-mask-detector/master/assets/model-json/model.json')

    printOutput('Successfully loaded model');

    liveDetect();
}


app();


