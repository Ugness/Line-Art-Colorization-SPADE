
let canvas, ctx,
    isCanvasSketch = true,
    brush = {
        x: 0,
        y: 0,
        color: '#000000',
        size: 10,
        down: false,
    },
    strokes = [],
    strokesFromOther = [],
    currentStroke = null;

function redraw () {
    ctx.clearRect(0, 0, canvas[0].width, canvas[0].height);
    ctx.lineCap = 'round';
    for (var i = 0; i < strokes.length; i++) {
        var s = strokes[i];
        ctx.strokeStyle = s.color;
        ctx.lineWidth = s.size;
        ctx.beginPath();
        ctx.moveTo(s.points[0].x, s.points[0].y);
        for (var j = 0; j < s.points.length; j++) {
            var p = s.points[j];
            ctx.lineTo(p.x, p.y);
        }
        ctx.stroke();
    }
}

function init () {
    $('#hint').attr({
        width: innerWidth < innerHeight ? innerWidth : innerHeight,
        height: innerWidth < innerHeight ? innerWidth : innerHeight,
    });

    canvas = $('#sketch');
    canvas.attr({
        width: innerWidth < innerHeight ? innerWidth : innerHeight,
        height: innerWidth < innerHeight ? innerWidth : innerHeight,
    });

    ctx = canvas[0].getContext('2d');

    function mouseEvent (e) {
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
        brush.down = true;
        currentStroke = {
            color: brush.color,
            size: brush.size,
            points: [],
        };
        strokes.push(currentStroke);
        mouseEvent(e);
    }
    function mouseUp (e) {
        brush.down = false;
        mouseEvent(e);
        currentStroke = null;
    }
    function mouseMove (e) {
        if (brush.down)
            mouseEvent(e);
    }

    canvas.mousedown(mouseDown)
        .mouseup(mouseUp)
        .mousemove(mouseMove);


    $('#save-btn').click(function () {
        window.open(canvas[0].toDataURL());
    });

    $('#undo-btn').click(function () {
        strokes.pop();
        redraw();
    });

    $('#clear-btn').click(function () {
        strokes = [];
        redraw();
    });

    function switchCanvas(isTargetSketch) {
        if (isTargetSketch === isCanvasSketch) {
            return
        }

        canvas.css('zIndex', '2');

        let nextCanvas;
        if (isCanvasSketch) {
            nextCanvas = $('#hint');
        }
        else {
            nextCanvas = $('#sketch');
        }

        canvas = nextCanvas;
        isCanvasSketch = !isCanvasSketch;

        canvas.css('zIndex', '3');
        ctx = canvas[0].getContext('2d');

        let temp = strokes;
        strokes = strokesFromOther;
        strokesFromOther = temp;

        canvas.mousedown(mouseDown)
            .mouseup(mouseUp)
            .mousemove(mouseMove);

        redraw()
    }

    $('#toSketch').click(function () {
        $('#toHint').removeClass('selected');
        $('#toSketch').addClass('selected');

        switchCanvas(true)
    });

    $('#toSketch').click();

    $('#toHint').click(function () {
        $('#toSketch').removeClass('selected');
        $('#toHint').addClass('selected');

        switchCanvas(false)
    });

    $('#color-picker').on('input', function () {
        brush.color = this.value;
    });

    $('#brush-size').on('input', function () {
        brush.size = this.value;
    });

}

$(init);