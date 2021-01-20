
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


function printOutput(msg) {
    $('#output').append(msg + '<br>');
}

function clearOutput() {
    $('#output').html('');
}

async function app() {
    $('#live-detector').hide();
    printOutput('Loading face-mask model..');


    // Load the model.
    model = await tf.loadLayersModel('https://raw.githubusercontent.com/itamarco/face-mask-detector/master/assets/model-json/model.json')

    printOutput('Successfully loaded face-mask model');

    $('#loading-gif').hide();
    $('#live-detector').show();
}


app();

