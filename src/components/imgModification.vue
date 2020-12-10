<template>
  <v-container fluid v-if="model1Org!==null">
    <v-row>
      <v-col>
        <div v-if="model1Prd === null">Model1:</div>
        <div v-else>Model1 (Epoch: {{ epoch }}. Prediction: {{ model1Prd }}):</div>
      </v-col>
    </v-row>
    <v-row>
      <v-col>
        <div>Ground-Truth: {{ label }}</div>
        <v-img :src="model1Org" width="100" contain></v-img>
      </v-col>
      <v-col v-for="act1 in model1Activations" :key="act1.id">
        <div>{{ act1.name }}</div>
        <v-img :src="act1.img" width="100" contain></v-img>
      </v-col>
    </v-row>
    <v-row>
      <v-col>
        <div v-if="model2Prd === null">Model2:</div>
        <div v-else>Model2 (Epoch: {{ epoch }}. Prediction: {{ model2Prd }}):</div>
      </v-col>
    </v-row>
    <v-row>
      <v-col>
        <div>Ground-Truth: {{ label }}</div>
        <v-img :src="model2Org" width="100" contain></v-img>
      </v-col>
      <v-col v-for="act2 in model2Activations" :key="act2.id">
        <div>{{ act2.name }}</div>
        <v-img :src="act2.img" width="100" contain></v-img>
      </v-col>
    </v-row>
  </v-container>
</template>

<script>
export default {
  name: "imgModification",

  data: () => ({
    model1Org: null,
    model2Org: null,
    label: null,
    model1Prd: null,
    model2Prd: null,
    epoch: null,
    model1Activations: [],
    model2Activations: []
  }),
  mounted() {
    this.$options.sockets.onmessage = (res) => {
      res = res.data;
      if (res.indexOf("sample_img***") !== -1) {
        this.model1Activations = []
        this.model2Activations = []
        res = res.split("***");
        this.model1Org = "data:image/png;base64," + res[1];
        this.model2Org = "data:image/png;base64," + res[1];
        this.label = res[2]
      } else if (res.indexOf("model1Activations***") !== -1) {
        this.model1Activations = []
        res = res.split("***");
        const imgLength = parseInt(res[1])
        this.model1Prd = res[2]
        this.epoch = res[3]
        let i = 0
        for (i = 0; i < imgLength; i++) {
          this.model1Activations.push({ id: i, name: res[4 + 2 * i], img: 'data:image/png;base64,' + res[5 + 2 * i] })
        }
      } else if (res.indexOf("model2Activations***") !== -1) {
        this.model2Activations = []
        res = res.split("***");
        const imgLength = parseInt(res[1])
        this.model2Prd = res[2]
        let i = 0
        for (i = 0; i < imgLength; i++) {
          this.model2Activations.push({ id: i, name: res[4 + 2 * i], img: 'data:image/png;base64,' + res[5 + 2 * i] })
        }
      }
    };
  },
};
</script>
