<template>
    <v-container fluid>
        <p>
            Canvas
        </p>
        <canvas id="canvas" width = 100 height = 100 @mousedown="startPainting" @mouseup="finishedPainting" @mousemove="draw"></canvas>
    </v-container>
</template>

<script>
export default {
    name: 'drawing',
    //data() {
    //    painting: false;
    //    ctx: null;
    //    canvas: null;
    //},

    data: () => ({
        painting: false,
        ctx: null,
        canvas: null,
        rect: null
    }),

    mounted() {
        this.canvas = document.getElementById("canvas");
        this.ctx = this.canvas.getContext("2d");
        this.canvas.height = 300;
        this.canvas.width = 300;
        this.rect = this.canvas.getBoundingClientRect();
        //this.ctx.fillRect(10,10,30,30)
        //this.vueCanvas = ctx;
    },

    methods: {
        startPainting(e) {
            this.painting = true;
            console.log(this.painting)
            this.draw(e)
        },
        finishedPainting() {
            this.painting = false;
            console.log(this.painting);
            this.ctx.beginPath()
        },
        draw(e) {
            if(!this.painting) return

            console.log("Mouse at " + e.clientX + ", " + e.clientY)

            this.ctx.lineWidth = 10;
            this.ctx.lineCap ="round";

            var x = e.clientX - this.rect.left;
            var y = e.clientY - this.rect.top;
            
            this.ctx.lineTo(x, y)
            this.ctx.stroke()

            this.ctx.beginPath()
            this.ctx.moveTo(x, y)

            //this.ctx.fillRect(e.clientX,e.clientY,30,30)
            //this.ctx.fillRect(10, 10,30,30)
            //console.log("Box")
        }
    }
}
</script>