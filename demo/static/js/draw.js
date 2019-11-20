
var canvas, ctx,
    brush = {
        x: 0,
        y: 0,
        color: '#000000',
        size: 10,
        down: false,
    },
    strokes = [],
    currentStroke = null;

function redraw () {
    ctx.clearRect(0, 0, canvas.width(), canvas.height());
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
    canvas = $('#draw');
    canvas.attr({
        width: window.innerWidth,
        height: window.innerHeight,
    });
    ctx = canvas[0].getContext('2d');

    function mouseEvent (e) {
        brush.x = e.pageX;
        brush.y = e.pageY;

        currentStroke.points.push({
            x: brush.x,
            y: brush.y,
        });

        redraw();
    }

    canvas.mousedown(function (e) {
        brush.down = true;

        currentStroke = {
            color: brush.color,
            size: brush.size,
            points: [],
        };

        strokes.push(currentStroke);

        mouseEvent(e);
    }).mouseup(function (e) {
        brush.down = false;

        mouseEvent(e);

        currentStroke = null;
    }).mousemove(function (e) {
        if (brush.down)
            mouseEvent(e);
    });

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

    $('#color-picker').on('input', function () {
        brush.color = this.value;
    });

    $('#brush-size').on('input', function () {
        brush.size = this.value;
    });

    $('#col-btn').click(function(){
        $.ajax({
            url: '/colorization/',
            data: {
                    'rgba': canvas[0].toDataURL(),
                    'width': canvas.width(),
                    'height': canvas.height()},
            method: 'POST',
            success: function(data) {
                var img = new Image;
                img.src = data['output'];
                ctx.drawImage(img,0,0);
            }
        });
    });
}

$(init);