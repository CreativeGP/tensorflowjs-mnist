function crep (limit, proc) {
    let i = 0;
    while (i < limit) {
        proc(i);
        i += 1;
    }
}

const tf = require('@tensorflow/tfjs');

const config = { strokeWeight: 8 };

let mouse = false;
let lastMousePos = { x: 0, y: 0 };

tf.loadModel("model/model.json")
    .then(model => {
        const ctx = document.getElementById('gaf').getContext('2d');
        ctx.lineWidth = config.strokeWeight;
        ctx.lineCap = 'square';

        document.getElementById('gaf').addEventListener('mousedown', e => mouse = true);
        document.getElementById('gaf').addEventListener('mouseup', e => mouse = false);
        document.getElementById('gaf').addEventListener('mouseout', e => mouse = false);
        document.getElementById('gaf').addEventListener('mousemove', e => {
            if (mouse) {
                ctx.beginPath();
                ctx.moveTo(lastMousePos.x, lastMousePos.y);
                ctx.lineTo(e.layerX, e.layerY);
                ctx.stroke();
            }
            lastMousePos = { x: e.layerX, y: e.layerY };
        });

        document.getElementById('go').addEventListener('mousedown', e => {
            document.getElementById('predic').innerHTML = "";
            predict();
            document.getElementById('gaf').getContext('2d').clearRect(0, 0, 200, 200);
        });

        function predict () {
            let oldimgdata = ctx.getImageData(0, 0, 200, 200);

            let newcvs = document.createElement('canvas');
            newcvs.width = 200;
            newcvs.height = 200;
            newcvs.getContext('2d').putImageData(oldimgdata, 0, 0);

            let dstcvs = document.createElement('canvas');
            dstcvs.width = 28;
            dstcvs.height = 28;
            dstcvs.getContext('2d').scale(0.14, 0.14);
            dstcvs.getContext('2d').drawImage(newcvs, 0, 0);
            document.body.appendChild(dstcvs);

            let imgdata = dstcvs.getContext('2d').getImageData(0, 0, 28, 28);


            let monodata = [];
            // monochrome
            for (let i=0, len = imgdata.data.length/4; i < len; i += 1) {
                monodata.push(imgdata.data[i*4+3]);
                monodata.push(0);
                monodata.push(0);
                monodata.push(0);
            }
            let monoimgdata = new ImageData(new Uint8ClampedArray(monodata), 28, 28);
//            let monoimgdata = imgdata;

            let input = tf.fromPixels(monoimgdata, 1).reshape([1, 28, 28, 1]).cast('float32').div(tf.scalar(255));
//            input = tf;
            const predict = model.predict(input).dataSync();

            let maxer = 0;
            crep(10, i => {
                document.getElementById('predic').innerHTML += `<h4 id="predic${i}">${i}: ${predict[i].toFixed(8)}</br></h4>`;
                maxer = predict.indexOf(Math.max(predict[i], predict[maxer]));
            });
            document.getElementById(`predic${maxer}`).style.backgroundColor = 'red';
        }
    });
