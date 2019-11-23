const canvasCount = 2;
let canvasArray = Array(canvasCount),
    currentCanvas = {
        Value: 0
    },
    brush = {
        x: 0,
        y: 0,
        color: '#000000',
        size: 10,
        down: false,
    },
    latent_shift = {
        value: 0.0,
    },
    isDeter = 0,
    currentStroke = null,
    sketchImage = null,
    colorImage = null,
    colorCanvas = null;

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

    colorCanvas = {
        canvas: $('#result'),
        ctx: null,
    };

    colorCanvas.canvas.attr({
        width: innerWidth < innerHeight ? innerWidth : innerHeight,
        height: innerWidth < innerHeight ? innerWidth : innerHeight,
    });
    colorCanvas.ctx = colorCanvas.canvas[0].getContext('2d');

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

    function touchEvent (e) {
        let canvas = canvasArray[currentCanvas.Value].canvas;

        let rect = canvas[0].getBoundingClientRect(),
            scaleX = canvas[0].width / rect.width,
            scaleY = canvas[0].height / rect.height;
        brush.x = (e.changedTouches[0].pageX - rect.left) * scaleX;
        brush.y = (e.changedTouches[0].pageY - rect.top) * scaleY;

        currentStroke.points.push({
            x: brush.x,
            y: brush.y,
        });

        redraw();
    }
    function touchStart (e) {
        console.log(currentCanvas.Value);
        brush.down = true;
        currentStroke = {
            color: brush.color,
            size: brush.size,
            points: [],
        };
        console.log("stroke added to %d\n", currentCanvas.Value);
        canvasArray[currentCanvas.Value].strokes.push(currentStroke);
        touchEvent(e);
    }
    function touchEnd (e) {
        brush.down = false;
        if (currentCanvas.Value === 0)
            touchEvent(e);
        currentStroke = null;
    }
    function touchMove (e) {
        if (brush.down && currentCanvas.Value === 0)
            touchEvent(e);
    }

    function addEventListeners() {
        canvasArray[currentCanvas.Value].canvas[0].addEventListener("mousedown", mouseDown);
        canvasArray[currentCanvas.Value].canvas[0].addEventListener("mouseup", mouseUp);
        canvasArray[currentCanvas.Value].canvas[0].addEventListener("mousemove", mouseMove);
        canvasArray[currentCanvas.Value].canvas[0].addEventListener("touchstart", touchStart);
        canvasArray[currentCanvas.Value].canvas[0].addEventListener("touchend", touchEnd);
        canvasArray[currentCanvas.Value].canvas[0].addEventListener("touchcancel", touchEnd);
        canvasArray[currentCanvas.Value].canvas[0].addEventListener("touchmove", touchMove);
        canvasArray[currentCanvas.Value].canvas[0].style.pointerEvents = "auto";
    }

    function removeEventListeners() {
        canvasArray[currentCanvas.Value].canvas[0].removeEventListener("mousedown", mouseDown);
        canvasArray[currentCanvas.Value].canvas[0].removeEventListener("mouseup", mouseUp);
        canvasArray[currentCanvas.Value].canvas[0].removeEventListener("mousemove", mouseMove);
        canvasArray[currentCanvas.Value].canvas[0].removeEventListener("touchstart", touchStart);
        canvasArray[currentCanvas.Value].canvas[0].removeEventListener("touchend", touchEnd);
        canvasArray[currentCanvas.Value].canvas[0].removeEventListener("touchcancel", touchEnd);
        canvasArray[currentCanvas.Value].canvas[0].removeEventListener("touchmove", touchMove);
        canvasArray[currentCanvas.Value].canvas[0].style.pointerEvents = "none";
    }

    addEventListeners();

    $('#load-btn').click(function(){
        $('#files').click();
    });

    $('#save-btn').click(function () {
        let element = document.createElement('a');
        element.href = colorCanvas.canvas[0].toDataURL();
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
        $('#files').val("");
        canvasArray[currentCanvas.Value].strokes = [];
        if (currentCanvas.Value === 0) sketchImage = null;
        redraw();
    });

    function switchCanvas(targetCanvas) {
        removeEventListeners();
        currentCanvas.Value = targetCanvas;
        addEventListeners();
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

    $('#latent-shift').on('input', function(){
        latent_shift.value = this.value;
    });

    $('#is-deter').on('input', function(){
        isDeter = this.value;
    });

    $('#col-btn').click(function(){
        $.ajax({
            url: '/colorization/',
            data: {
                    'rgba': canvasArray[1].canvas[0].toDataURL(),
                    'width': canvasArray[1].canvas[0].width,
                    'height': canvasArray[1].canvas[0].height,
                    'line': canvasArray[0].canvas[0].toDataURL(),
                    'z': latent_shift.value,
                    'isDeter': isDeter},
            method: 'POST',
            success: function(data) {
                console.log('color data came!');
                colorImage = new Image;
                colorImage.src = data['output'];
                colorImage.onload = function(){
                  scaletoFit(this, colorCanvas.canvas[0], colorCanvas.ctx);
                };
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
                canvasArray[0].strokes = [];
                sketchImage = new Image;
                sketchImage.src = data['output'];
                sketchImage.onload = redraw
            }
        });
    });
}

function scaletoFit(img, canvas, ctx){
    let scale = Math.min(canvas.width / img.width, canvas.height / img.height);
    let x = (canvas.width / 2) - (img.width / 2) * scale;
    let y = (canvas.height / 2) - (img.height / 2) * scale;
    console.log(canvas.width,canvas.height,img.width,img.height);
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
