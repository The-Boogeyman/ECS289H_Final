<template>
  <v-container fluid>
    <v-row>
      <v-col v-if="model1Org!==null">
        <p>Model1:</p>
        <v-img :src="model1Org" width="100" contain></v-img>
      </v-col>
    </v-row>
    <v-row>
      <v-col v-if="model2Org!==null">
        <p>Model2:</p>
        <v-img :src="model2Org" width="100" contain></v-img>
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
  }),
  mounted() {
    this.$options.sockets.onmessage = (res) => {
      res = res.data;
      if (res.indexOf("sample_img***") !== -1) {
        res = res.split("***");
        this.model1Org = "data:image/png;base64," + res[1];
        this.model2Org = "data:image/png;base64," + res[1];
      }
    };
  },
};
</script>
