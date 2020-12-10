<template>
  <div id="trainAcc" style="width:300px; height: 200px" class="echarts"></div>
</template>

<script>
import echarts from "echarts";

export default {
  name: "plotAccTrain",

  data: () => ({
    id: "trainAcc",
    chart: null,
    option: {
      xAxis: {
        name: "Epoch",
        data: [],
      },
      yAxis: {
        type: "value",
        name: "Train Accuracy(%)",
      },
      legend: {
        data: ["model1", "model2"],
      },
      grid: {
        left: 50,
        right: 50,
        top: 50,
        bottom: 50
      },
      series: [
        {
          name: "model1",
          type: "line",
          data: [],
        },
        {
          name: "model2",
          type: "line",
          data: [],
        },
      ],
    },
  }),
  methods: {
    init() {
      this.chart = echarts.init(document.getElementById(this.id));
      this.chart.setOption(this.option);
    },
    updateValue(res1, res2, res3) {
      this.option.xAxis.data = res1;
      this.option.series[0].data = res2;
      this.option.series[1].data = res3;
      this.chart.setOption(this.option);
    },
  },
  mounted () {
    this.init()
    this.updateValue([1, 2], [50.1, 58.1], [30.8, 61.5])
  }
};
</script>
