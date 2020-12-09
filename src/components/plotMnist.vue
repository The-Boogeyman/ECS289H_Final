<template>
  <div id="mnist" style="width:700px; height: 400px" class="echarts"></div>
</template>

<script>
import echarts from "echarts";

export default {
  name: "plotMnist",

  data: () => ({
    id: "mnist",
    chart: null,
    option: {
      xAxis: {
        type: 'value',
        scale: true,
        show: false,
        splitLine: {
          show: false
        },
        axisLine: {
          show: false
        },
        axisTick: {
          show: false
        }
      },
      yAxis: {
        type: 'value',
        scale: true,
        show: false,
        splitLine: {
          show: false
        },
        axisLine: {
          show: false
        },
        axisTick: {
          show: false
        }
      },
      legend: {
        data: ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
      },
      grid: {
        left: 50,
        right: 50,
        top: 50,
        bottom: 50
      },
      series: [
        {
          name: "1",
          type: "scatter",
          data: [],
        },
        {
          name: "2",
          type: "scatter",
          data: [],
        },
        {
          name: "3",
          type: "scatter",
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
    updateValue(res0, res1, res2) {
      this.option.series[0].data = res0;
      this.option.series[1].data = res1;
      this.option.series[2].data = res2;
      this.chart.setOption(this.option);
    },
  },
  mounted () {
    this.init()
    this.updateValue([[0, 0], [1, 1]], [[2, 3], [4, 6]], [[7, 7], [9, 9], [10, 10]])
  }
};
</script>
