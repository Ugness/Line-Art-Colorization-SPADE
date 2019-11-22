const canvasCount = 2;
let canvasArray = Array(canvasCount),
    currentCanvas = {Value: 0},
    brush = {
        x: 0,
        y: 0,
        color: '#000000',
        size: 10,
        down: false,
    },
    z_vector = {
        value: 0.0,
    },
    currentStroke = null,
    sketchImage = null;

function redraw () {
    for (let c = 0 ; c < canvasCount ; c++) {
        let ctx = canvasArray[c].ctx;
        let w = canvasArray[c].canvas[0].width;
        let strokes = canvasArray[c].strokes;

        if(c === 1) {
            ctx.clearRect(0, 0, w, w);
        }
        else{
            let old_style = ctx.fillStyle;
            ctx.fillStyle = 'white';
            ctx.fillRect(0,0,w,w);
            ctx.fillStyle = old_style;
        }
        ctx.lineCap = 'square';
        if (sketchImage != null && c === 0) {
            console.log("sketch loaded?");
            scaletoFit(sketchImage, canvasArray[c].canvas[0], ctx);
        }

        for (let i = 0; i < strokes.length; i++) {
            let s = strokes[i];
            ctx.strokeStyle = s.color;
            ctx.lineWidth = s.size;
            ctx.beginPath();
            ctx.moveTo(s.points[0].x, s.points[0].y);
            for (let j = 0; j < s.points.length; j++) {
                let p = s.points[j];
                ctx.lineTo(p.x, p.y);
            }
            ctx.stroke();
        }
    }
}

function init () {
    canvasArray[0] = {
        canvas: $('#sketch'),
        ctx: null,
        strokes: [],
    };
    canvasArray[0].canvas.attr({
        width: innerWidth < innerHeight ? innerWidth : innerHeight,
        height: innerWidth < innerHeight ? innerWidth : innerHeight,
    });
    canvasArray[0].ctx = canvasArray[0].canvas[0].getContext('2d');

    canvasArray[1] = {
        canvas: $('#hint'),
        ctx: null,
        strokes: [],
    };
    canvasArray[1].canvas.attr({
        width: innerWidth < innerHeight ? innerWidth : innerHeight,
        height: innerWidth < innerHeight ? innerWidth : innerHeight,
    });
    canvasArray[1].ctx = canvasArray[1].canvas[0].getContext('2d');

    $('#result').attr({
        width: innerWidth < innerHeight ? innerWidth : innerHeight,
        height: innerWidth < innerHeight ? innerWidth : innerHeight,
    });

    function mouseEvent (e) {
        let canvas = canvasArray[currentCanvas.Value].canvas;

        let rect = canvas[0].getBoundingClientRect(),
            scaleX = canvas[0].width / rect.width,
            scaleY = canvas[0].height / rect.height;
        brush.x = (e.pageX - rect.left) * scaleX;
        brush.y = (e.pageY - rect.top) * scaleY;

        currentStroke.points.push({
            x: brush.x,
            y: brush.y,
        });

        redraw();
    }
    function mouseDown (e) {
        console.log(currentCanvas.Value);
        brush.down = true;
        currentStroke = {
            color: brush.color,
            size: brush.size,
            points: [],
        };
        console.log("stroke added to %d\n", currentCanvas.Value);
        canvasArray[currentCanvas.Value].strokes.push(currentStroke);
        mouseEvent(e);
    }
    function mouseUp (e) {
        brush.down = false;
        if (currentCanvas.Value === 0)
            mouseEvent(e);
        currentStroke = null;
    }
    function mouseMove (e) {
        if (brush.down && currentCanvas.Value === 0)
            mouseEvent(e);
    }

    canvasArray[currentCanvas.Value].canvas[0].addEventListener("mousedown", mouseDown);
    canvasArray[currentCanvas.Value].canvas[0].addEventListener("mouseup", mouseUp);
    canvasArray[currentCanvas.Value].canvas[0].addEventListener("mousemove", mouseMove);

    $('#load-btn').click(function(){
        $('#files').click();
    });

    $('#save-btn').click(function () {
        var element = document.createElement('a');
        element.href = $('#result')[0].toDataURL();
        element.download = "result.png";
        element.style.display='none';
        document.body.appendChild(element);
        element.click();
        document.body.removeChild(element);
    });

    $('#undo-btn').click(function () {
        console.log("Undo: %d\n",currentCanvas.Value);
        canvasArray[currentCanvas.Value].strokes.pop();
        redraw();
    });

    $('#clear-btn').click(function () {
        canvasArray[currentCanvas.Value].strokes = [];
        if (currentCanvas.Value === 0) sketchImage = null;
        redraw();
    });

    function switchCanvas(targetCanvas) {
        canvasArray[currentCanvas.Value].canvas[0].removeEventListener("mousedown", mouseDown);
        canvasArray[currentCanvas.Value].canvas[0].removeEventListener("mouseup", mouseUp);
        canvasArray[currentCanvas.Value].canvas[0].removeEventListener("mousemove", mouseMove);
        canvasArray[currentCanvas.Value].canvas[0].style.pointerEvents = "none";


        currentCanvas.Value = targetCanvas;
        console.log(currentCanvas.Value);

        canvasArray[currentCanvas.Value].canvas[0].addEventListener("mousedown", mouseDown);
        canvasArray[currentCanvas.Value].canvas[0].addEventListener("mouseup", mouseUp);
        canvasArray[currentCanvas.Value].canvas[0].addEventListener("mousemove", mouseMove);
        canvasArray[currentCanvas.Value].canvas[0].style.pointerEvents = "auto";

        redraw();
    }

    $('#toSketch').click(function () {
        $('#toHint').removeClass('selected');
        $('#toSketch').addClass('selected');

        switchCanvas(0)
    });

    $('#toSketch').click();

    $('#toHint').click(function () {
        $('#toSketch').removeClass('selected');
        $('#toHint').addClass('selected');

        switchCanvas(1)
    });

    $('#color-picker').on('input', function () {
        brush.color = this.value;
    });

    $('#brush-size').on('input', function () {
        brush.size = this.value;
    });

    $('#z-vector').on('input', function(){
        z_vector.value = this.value;
    });

    $('#col-btn').click(function(){
        $.ajax({
            url: '/colorization/',
            data: {
                    'rgba': canvasArray[1].canvas[0].toDataURL(),
                    'width': canvasArray[1].canvas[0].width,
                    'height': canvasArray[1].canvas[0].height,
                    'line': canvasArray[0].canvas[0].toDataURL(),
                    'z': z_vector.value},
            method: 'POST',
            success: function(data) {
                let temp_canvas = $('#result')[0];
                let temp_ctx = temp_canvas.getContext('2d');
                let w = temp_canvas.width;
                temp_ctx.clearRect(0, 0, w, w);
                let img = new Image;
                img.src = data['output'];
                scaletoFit(img, temp_canvas, temp_ctx);
            }
        });
    });

    $('#sim-btn').click(function(){
        $.ajax({
            url: '/simplification/',
            data: {
                    'line': canvasArray[0].canvas[0].toDataURL(),
                    'width': canvasArray[0].canvas[0].width,
                    'height': canvasArray[0].canvas[0].height},
            method: 'POST',
            success: function(data) {
                sketchImage = new Image;
                sketchImage.src = data['output'];

                let w = canvasArray[0].canvas[0].width;
                canvasArray[0].strokes = [];
                redraw();
            }
        });
    });
}

function scaletoFit(img, canvas, ctx){
    var scale = Math.min(canvas.width / img.width, canvas.height / img.height);
    var x = (canvas.width / 2) - (img.width / 2) * scale;
    var y = (canvas.height / 2) - (img.height / 2) * scale;
    ctx.drawImage(img, x, y, img.width * scale, img.height * scale);
}

function handleFileSelect(evt) {
    var files = evt.target.files; // FileList object

    // Loop through the FileList and render image files as thumbnails.
    var reader = new FileReader();
    f = files[0];
      // Closure to capture the file information.
      reader.onload = (function() {
          return function(e) {
              sketchImage = new Image;
              sketchImage.src = e.target.result;
              sketchImage.onload = redraw;
          };
      })(f);

      // Read in the image file as a data URL.
    reader.readAsDataURL(f);
  }

$(init);
document.getElementById('files').addEventListener('change', handleFileSelect, false);
