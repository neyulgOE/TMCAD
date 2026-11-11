# CAD-Native 6DoF Table Estimator V1

100% CAD-native 6DoF pose estimator for Orange Pi 5B / RK3588S.

- adapter: `cad_hierarchy_v1`
- task_type: `cad_6dof`
- object_type: `table`
- application: `cad_pose`
- cad_model: `TABLE-DEMO`
- input: `(1, 7, 1024, 20)` float32 graph features from depth
- output: `(1, 128, 1024)` float32 point descriptors matching 512 CAD centroids
