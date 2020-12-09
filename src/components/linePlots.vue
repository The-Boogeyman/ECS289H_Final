<template>
  <v-container fluid>
    <v-row no-gutters>
      <v-col cols="auto">
        <div id="loss" style="width:300px; height: 200px" class="echarts"></div>
      </v-col>
    </v-row>
    <v-row no-gutters>
      <v-col cols="auto">
        <div id="acc" style="width:300px; height: 200px" class="echarts"></div>
      </v-col>
    </v-row>
  </v-container>
</template>

<script>
import echarts from "echarts";

export default {
  data: () => ({
    chart: null,
  }),
  methods: {
    init() {
      this.chart = echarts.init(document.getElementById(this.id));
    },
  },
};
</script>
