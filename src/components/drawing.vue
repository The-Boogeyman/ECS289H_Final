<template>
  <v-container fluid>
    <v-row>
      <v-col>
        <p>Canvas</p>
        <canvas
          id="canvas"
          width="300"
          height="300"
          @mousedown="startPainting"
          @mouseup="finishedPainting"
          @mousemove="draw"
        ></canvas>
      </v-col>
    </v-row>
  </v-container>
</template>

<script>
export default {
  name: "drawing",
  //data() {
  //    painting: false;
  //    ctx: null;
  //    canvas: null;
  //},
  data: () => ({
    painting: false,
    ctx: null,
    canvas: null,
    rect: null,
  }),
  mounted() {
    this.canvas = document.getElementById("canvas");
    this.ctx = this.canvas.getContext("2d");
    this.canvas.height = 300;
    this.canvas.width = 300;
    this.rect = this.canvas.getBoundingClientRect();
    //this.ctx.fillRect(10,10,30,30)
    //this.vueCanvas = ctx;
    this.$options.sockets.onmessage = (res) => {
      res = res.data;
      if (res.indexOf("sample_canvas***") !== -1) {
        res = res.split("***");
        var img = new Image();
        img.src = "data:image/png;base64," + res[1];
        img.onload = () => {
          console.log("this.ctx:", this.ctx);
          this.ctx.drawImage(img, 0, 0, 300, 300);
        };
      }
    };
  },
  methods: {
    startPainting(e) {
      this.painting = true;
        console.log(this.painting);
      this.draw(e);
    },
    finishedPainting() {
      this.painting = false;
        console.log(this.painting);
      this.ctx.beginPath();
    },
    draw(e) {
      if (!this.painting) return;
        console.log("Mouse at " + e.clientX + ", " + e.clientY);
      this.ctx.lineWidth = 10;
      this.ctx.lineCap = "round";
      var x = e.clientX - this.rect.left;
      var y = e.clientY - this.rect.top;

      this.ctx.lineTo(x, y);
      this.ctx.stroke();
      this.ctx.beginPath();
      this.ctx.moveTo(x, y);
      //this.ctx.fillRect(e.clientX,e.clientY,30,30)
      //this.ctx.fillRect(10, 10,30,30)
      //console.log("Box")
    },
  },
};
</script>
