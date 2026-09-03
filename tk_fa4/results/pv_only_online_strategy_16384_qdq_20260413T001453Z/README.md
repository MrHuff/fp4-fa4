# Online P-Strategy `S=16384` Addendum

- Input mode: `random_live_fp4`
- Execution mode: `qdq_proxy`
- Heads: `4`
- Strategies: `live_direct`, `live_sa3_baseline`, `live_localcta_cta_amax_experimental`

## Per Distribution

- `gaussian` / `localcta`: order `('live_direct', 'live_localcta_cta_amax_experimental', 'live_sa3_baseline')`
- `live_direct`: MAE `0.000607609`, norm-MAE `0.161555967`, RMSE `0.001244612`, norm-RMSE `0.330927167`
- `live_localcta_cta_amax_experimental`: MAE `0.000628152`, norm-MAE `0.167017916`, RMSE `0.001198684`, norm-RMSE `0.318715723`
- `live_sa3_baseline`: MAE `0.000638443`, norm-MAE `0.169754183`, RMSE `0.001217073`, norm-RMSE `0.323604949`
- `gaussian` / `mxfp4_v3`: order `('live_sa3_baseline', 'live_localcta_cta_amax_experimental', 'live_direct')`
- `live_sa3_baseline`: MAE `0.000630536`, norm-MAE `0.167651935`, RMSE `0.001595544`, norm-RMSE `0.424235913`
- `live_localcta_cta_amax_experimental`: MAE `0.000702410`, norm-MAE `0.186762210`, RMSE `0.001635509`, norm-RMSE `0.434862188`
- `live_direct`: MAE `0.000730522`, norm-MAE `0.194236859`, RMSE `0.001357809`, norm-RMSE `0.361024970`

- `laplace` / `localcta`: order `('live_direct', 'live_localcta_cta_amax_experimental', 'live_sa3_baseline')`
- `live_direct`: MAE `0.000596156`, norm-MAE `0.157150337`, RMSE `0.001227177`, norm-RMSE `0.323491114`
- `live_localcta_cta_amax_experimental`: MAE `0.000608908`, norm-MAE `0.160511814`, RMSE `0.001165569`, norm-RMSE `0.307250994`
- `live_sa3_baseline`: MAE `0.000618030`, norm-MAE `0.162916293`, RMSE `0.001177769`, norm-RMSE `0.310467049`
- `laplace` / `mxfp4_v3`: order `('live_sa3_baseline', 'live_localcta_cta_amax_experimental', 'live_direct')`
- `live_sa3_baseline`: MAE `0.000626688`, norm-MAE `0.165198605`, RMSE `0.001602239`, norm-RMSE `0.422359752`
- `live_localcta_cta_amax_experimental`: MAE `0.000664835`, norm-MAE `0.175254420`, RMSE `0.001635638`, norm-RMSE `0.431164003`
- `live_direct`: MAE `0.000722896`, norm-MAE `0.190559824`, RMSE `0.001386533`, norm-RMSE `0.365498433`

- `student_t3` / `localcta`: order `('live_direct', 'live_localcta_cta_amax_experimental', 'live_sa3_baseline')`
- `live_direct`: MAE `0.000576244`, norm-MAE `0.155900235`, RMSE `0.001268369`, norm-RMSE `0.343151557`
- `live_localcta_cta_amax_experimental`: MAE `0.000581943`, norm-MAE `0.157442025`, RMSE `0.001205867`, norm-RMSE `0.326241867`
- `live_sa3_baseline`: MAE `0.000592325`, norm-MAE `0.160250838`, RMSE `0.001208757`, norm-RMSE `0.327023585`
- `student_t3` / `mxfp4_v3`: order `('live_sa3_baseline', 'live_localcta_cta_amax_experimental', 'live_direct')`
- `live_sa3_baseline`: MAE `0.000722519`, norm-MAE `0.195474109`, RMSE `0.001975435`, norm-RMSE `0.534444966`
- `live_localcta_cta_amax_experimental`: MAE `0.000771437`, norm-MAE `0.208708843`, RMSE `0.002103587`, norm-RMSE `0.569115847`
- `live_direct`: MAE `0.000771919`, norm-MAE `0.208839250`, RMSE `0.001564364`, norm-RMSE `0.423231708`

- `student_t5` / `localcta`: order `('live_direct', 'live_localcta_cta_amax_experimental', 'live_sa3_baseline')`
- `live_direct`: MAE `0.000568182`, norm-MAE `0.152796184`, RMSE `0.001159005`, norm-RMSE `0.311681043`
- `live_localcta_cta_amax_experimental`: MAE `0.000593049`, norm-MAE `0.159483308`, RMSE `0.001123548`, norm-RMSE `0.302145918`
- `live_sa3_baseline`: MAE `0.000603333`, norm-MAE `0.162249037`, RMSE `0.001140268`, norm-RMSE `0.306642104`
- `student_t5` / `mxfp4_v3`: order `('live_sa3_baseline', 'live_localcta_cta_amax_experimental', 'live_direct')`
- `live_sa3_baseline`: MAE `0.000647080`, norm-MAE `0.174013456`, RMSE `0.001538106`, norm-RMSE `0.413629085`
- `live_localcta_cta_amax_experimental`: MAE `0.000657840`, norm-MAE `0.176907166`, RMSE `0.001512593`, norm-RMSE `0.406768156`
- `live_direct`: MAE `0.000741095`, norm-MAE `0.199295975`, RMSE `0.001372312`, norm-RMSE `0.369043755`

- `signed_lognormal` / `localcta`: order `('live_direct', 'live_localcta_cta_amax_experimental', 'live_sa3_baseline')`
- `live_direct`: MAE `0.000562157`, norm-MAE `0.151471097`, RMSE `0.001133544`, norm-RMSE `0.305429353`
- `live_localcta_cta_amax_experimental`: MAE `0.000604994`, norm-MAE `0.163013383`, RMSE `0.001124941`, norm-RMSE `0.303111292`
- `live_sa3_baseline`: MAE `0.000616163`, norm-MAE `0.166022782`, RMSE `0.001135417`, norm-RMSE `0.305933779`
- `signed_lognormal` / `mxfp4_v3`: order `('live_sa3_baseline', 'live_localcta_cta_amax_experimental', 'live_direct')`
- `live_sa3_baseline`: MAE `0.000688007`, norm-MAE `0.185381047`, RMSE `0.001546495`, norm-RMSE `0.416697480`
- `live_localcta_cta_amax_experimental`: MAE `0.000718300`, norm-MAE `0.193543277`, RMSE `0.001548130`, norm-RMSE `0.417137869`
- `live_direct`: MAE `0.000757578`, norm-MAE `0.204126606`, RMSE `0.001386339`, norm-RMSE `0.373543868`

- `gaussian_spikes` / `localcta`: order `('live_localcta_cta_amax_experimental', 'live_sa3_baseline', 'live_direct')`
- `live_localcta_cta_amax_experimental`: MAE `0.000609581`, norm-MAE `0.165725440`, RMSE `0.001174052`, norm-RMSE `0.319187015`
- `live_sa3_baseline`: MAE `0.000618552`, norm-MAE `0.168164294`, RMSE `0.001184496`, norm-RMSE `0.322026373`
- `live_direct`: MAE `0.000625919`, norm-MAE `0.170167190`, RMSE `0.001251865`, norm-RMSE `0.340341764`
- `gaussian_spikes` / `mxfp4_v3`: order `('live_sa3_baseline', 'live_localcta_cta_amax_experimental', 'live_direct')`
- `live_sa3_baseline`: MAE `0.000808235`, norm-MAE `0.219733214`, RMSE `0.001793856`, norm-RMSE `0.487691846`
- `live_localcta_cta_amax_experimental`: MAE `0.000832269`, norm-MAE `0.226267224`, RMSE `0.001753852`, norm-RMSE `0.476816163`
- `live_direct`: MAE `0.000838568`, norm-MAE `0.227979831`, RMSE `0.001544502`, norm-RMSE `0.419900595`
