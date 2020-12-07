<template>
  <v-container fluid>
    <v-row no-gutters>
      <v-col cols="2">
        <v-navigation-drawer permanent width="100%">
          <v-card flat>
            <v-list dense>
              <v-subheader>Model1</v-subheader>
              <v-list-item-group>
                <v-list-item>
                  <v-text-field
                    v-model="nLayers1"
                    label="# of layers"
                    type="number"
                  ></v-text-field>
                </v-list-item>
                <v-list-item>
                    <v-text-field
                      label="out size of each layer"
                      type="number"
                    ></v-text-field>
                  </v-list-item>
              </v-list-item-group>
            </v-list>
          </v-card>
        </v-navigation-drawer>
      </v-col>
    </v-row>
  </v-container>
</template>

<script>
export default {
  data: () => ({
    nLayers1: 1,
  }),
};
</script>
