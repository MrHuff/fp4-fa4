# Online P-Strategy Distribution Sweep

- Input mode: `random_live_fp4`
- Execution mode: `qdq_proxy`
- Heads: `8`
- Seqlens: `(2048, 4096, 8192)`
- Distributions: `(gaussian, laplace, student_t3, student_t5, signed_lognormal, gaussian_spikes)`
- Compared strategies: `live_direct`, `live_sa3_baseline`, `live_localcta_cta_amax_experimental`

## Same-Backend Aggregate

- `localcta` / `gaussian`: wins direct `1`, wins SA3 `0`, wins localCTA `2`, mean rank direct `2.333`, SA3 `2.333`, localCTA `1.333`, mean delta norm-MAE SA3-direct `+0.000284279`, localCTA-direct `-0.002300103`
- `localcta` / `laplace`: wins direct `1`, wins SA3 `0`, wins localCTA `2`, mean rank direct `2.000`, SA3 `2.667`, localCTA `1.333`, mean delta norm-MAE SA3-direct `+0.000616345`, localCTA-direct `-0.001892444`
- `localcta` / `student_t3`: wins direct `2`, wins SA3 `0`, wins localCTA `1`, mean rank direct `1.333`, SA3 `3.000`, localCTA `1.667`, mean delta norm-MAE SA3-direct `+0.003857771`, localCTA-direct `+0.000907710`
- `localcta` / `student_t5`: wins direct `1`, wins SA3 `0`, wins localCTA `2`, mean rank direct `2.333`, SA3 `2.333`, localCTA `1.333`, mean delta norm-MAE SA3-direct `-0.000209038`, localCTA-direct `-0.002769301`
- `localcta` / `signed_lognormal`: wins direct `2`, wins SA3 `0`, wins localCTA `1`, mean rank direct `1.333`, SA3 `3.000`, localCTA `1.667`, mean delta norm-MAE SA3-direct `+0.005713050`, localCTA-direct `+0.002823948`
- `localcta` / `gaussian_spikes`: wins direct `0`, wins SA3 `0`, wins localCTA `3`, mean rank direct `3.000`, SA3 `2.000`, localCTA `1.000`, mean delta norm-MAE SA3-direct `-0.006094667`, localCTA-direct `-0.008399188`
- `mxfp4_v3` / `gaussian`: wins direct `0`, wins SA3 `3`, wins localCTA `0`, mean rank direct `3.000`, SA3 `1.000`, localCTA `2.000`, mean delta norm-MAE SA3-direct `-0.015905278`, localCTA-direct `-0.006081938`
- `mxfp4_v3` / `laplace`: wins direct `0`, wins SA3 `3`, wins localCTA `0`, mean rank direct `3.000`, SA3 `1.000`, localCTA `2.000`, mean delta norm-MAE SA3-direct `-0.015781400`, localCTA-direct `-0.012488579`
- `mxfp4_v3` / `student_t3`: wins direct `0`, wins SA3 `3`, wins localCTA `0`, mean rank direct `2.333`, SA3 `1.000`, localCTA `2.667`, mean delta norm-MAE SA3-direct `-0.011462051`, localCTA-direct `+0.001290703`
- `mxfp4_v3` / `student_t5`: wins direct `0`, wins SA3 `1`, wins localCTA `2`, mean rank direct `3.000`, SA3 `1.667`, localCTA `1.333`, mean delta norm-MAE SA3-direct `-0.016791144`, localCTA-direct `-0.018790491`
- `mxfp4_v3` / `signed_lognormal`: wins direct `0`, wins SA3 `3`, wins localCTA `0`, mean rank direct `3.000`, SA3 `1.000`, localCTA `2.000`, mean delta norm-MAE SA3-direct `-0.012177009`, localCTA-direct `-0.006315773`
- `mxfp4_v3` / `gaussian_spikes`: wins direct `1`, wins SA3 `2`, wins localCTA `0`, mean rank direct `2.000`, SA3 `1.333`, localCTA `2.667`, mean delta norm-MAE SA3-direct `-0.002560310`, localCTA-direct `+0.001931253`

## Canonical Method Aggregate

- `mxfp4_online`: wins `0`, mean rank `3.000`, mean norm-MAE `0.203266514`
- `sa3_baseline`: wins `0`, mean rank `2.000`, mean norm-MAE `0.164754540`
- `localcta_cta_amax`: wins `18`, mean rank `1.000`, mean norm-MAE `0.162121687`

## Canonical Method Per Case

- `gaussian` / `S=2048`: winner `localcta_cta_amax`
- `mxfp4_online`: `live_direct/mxfp4_v3` MAE `0.002008345`, norm-MAE `0.191990926`, RMSE `0.003351770`, norm-RMSE `0.320417750`
- `sa3_baseline`: `live_sa3_baseline/localcta` MAE `0.001766264`, norm-MAE `0.168848853`, RMSE `0.003011428`, norm-RMSE `0.287882217`
- `localcta_cta_amax`: `live_localcta_cta_amax_experimental/localcta` MAE `0.001738438`, norm-MAE `0.166188699`, RMSE `0.002996230`, norm-RMSE `0.286429329`
- `gaussian` / `S=4096`: winner `localcta_cta_amax`
- `mxfp4_online`: `live_direct/mxfp4_v3` MAE `0.001431246`, norm-MAE `0.200145236`, RMSE `0.002486227`, norm-RMSE `0.347673551`
- `sa3_baseline`: `live_sa3_baseline/localcta` MAE `0.001233585`, norm-MAE `0.172504354`, RMSE `0.002208538`, norm-RMSE `0.308841586`
- `localcta_cta_amax`: `live_localcta_cta_amax_experimental/localcta` MAE `0.001215535`, norm-MAE `0.169980153`, RMSE `0.002192201`, norm-RMSE `0.306557074`
- `gaussian` / `S=8192`: winner `localcta_cta_amax`
- `mxfp4_online`: `live_direct/mxfp4_v3` MAE `0.001018671`, norm-MAE `0.192951899`, RMSE `0.001811921`, norm-RMSE `0.343205445`
- `sa3_baseline`: `live_sa3_baseline/localcta` MAE `0.000890398`, norm-MAE `0.168655039`, RMSE `0.001639452`, norm-RMSE `0.310537290`
- `localcta_cta_amax`: `live_localcta_cta_amax_experimental/localcta` MAE `0.000876837`, norm-MAE `0.166086248`, RMSE `0.001621322`, norm-RMSE `0.307103098`
- `laplace` / `S=2048`: winner `localcta_cta_amax`
- `mxfp4_online`: `live_direct/mxfp4_v3` MAE `0.002042236`, norm-MAE `0.193961379`, RMSE `0.003453538`, norm-RMSE `0.327999760`
- `sa3_baseline`: `live_sa3_baseline/localcta` MAE `0.001706172`, norm-MAE `0.162043718`, RMSE `0.002901435`, norm-RMSE `0.275563845`
- `localcta_cta_amax`: `live_localcta_cta_amax_experimental/localcta` MAE `0.001679796`, norm-MAE `0.159538637`, RMSE `0.002884985`, norm-RMSE `0.274001489`
- `laplace` / `S=4096`: winner `localcta_cta_amax`
- `mxfp4_online`: `live_direct/mxfp4_v3` MAE `0.001457778`, norm-MAE `0.196312980`, RMSE `0.002553057`, norm-RMSE `0.343809673`
- `sa3_baseline`: `live_sa3_baseline/localcta` MAE `0.001233579`, norm-MAE `0.166121023`, RMSE `0.002185637`, norm-RMSE `0.294330722`
- `localcta_cta_amax`: `live_localcta_cta_amax_experimental/localcta` MAE `0.001214448`, norm-MAE `0.163544652`, RMSE `0.002164908`, norm-RMSE `0.291539266`
- `laplace` / `S=8192`: winner `localcta_cta_amax`
- `mxfp4_online`: `live_direct/mxfp4_v3` MAE `0.001024469`, norm-MAE `0.195416537`, RMSE `0.001860543`, norm-RMSE `0.354896916`
- `sa3_baseline`: `live_sa3_baseline/localcta` MAE `0.000861840`, norm-MAE `0.164395288`, RMSE `0.001584255`, norm-RMSE `0.302195288`
- `localcta_cta_amax`: `live_localcta_cta_amax_experimental/localcta` MAE `0.000849023`, norm-MAE `0.161950372`, RMSE `0.001566440`, norm-RMSE `0.298797020`
- `student_t3` / `S=2048`: winner `localcta_cta_amax`
- `mxfp4_online`: `live_direct/mxfp4_v3` MAE `0.002095794`, norm-MAE `0.206307075`, RMSE `0.003573387`, norm-RMSE `0.351759335`
- `sa3_baseline`: `live_sa3_baseline/localcta` MAE `0.001643266`, norm-MAE `0.161760867`, RMSE `0.002847402`, norm-RMSE `0.280294425`
- `localcta_cta_amax`: `live_localcta_cta_amax_experimental/localcta` MAE `0.001612813`, norm-MAE `0.158763072`, RMSE `0.002810236`, norm-RMSE `0.276635795`
- `student_t3` / `S=4096`: winner `localcta_cta_amax`
- `mxfp4_online`: `live_direct/mxfp4_v3` MAE `0.001519538`, norm-MAE `0.210133747`, RMSE `0.002770202`, norm-RMSE `0.383085567`
- `sa3_baseline`: `live_sa3_baseline/localcta` MAE `0.001176319`, norm-MAE `0.162670791`, RMSE `0.002107990`, norm-RMSE `0.291509642`
- `localcta_cta_amax`: `live_localcta_cta_amax_experimental/localcta` MAE `0.001155397`, norm-MAE `0.159777476`, RMSE `0.002087274`, norm-RMSE `0.288644811`
- `student_t3` / `S=8192`: winner `localcta_cta_amax`
- `mxfp4_online`: `live_direct/mxfp4_v3` MAE `0.001104566`, norm-MAE `0.213853056`, RMSE `0.002045044`, norm-RMSE `0.395937546`
- `sa3_baseline`: `live_sa3_baseline/localcta` MAE `0.000846018`, norm-MAE `0.163796152`, RMSE `0.001581048`, norm-RMSE `0.306104103`
- `localcta_cta_amax`: `live_localcta_cta_amax_experimental/localcta` MAE `0.000830734`, norm-MAE `0.160837078`, RMSE `0.001573166`, norm-RMSE `0.304577987`
- `student_t5` / `S=2048`: winner `localcta_cta_amax`
- `mxfp4_online`: `live_direct/mxfp4_v3` MAE `0.002006725`, norm-MAE `0.194129817`, RMSE `0.003414048`, norm-RMSE `0.330273724`
- `sa3_baseline`: `live_sa3_baseline/localcta` MAE `0.001678327`, norm-MAE `0.162360733`, RMSE `0.002882304`, norm-RMSE `0.278832980`
- `localcta_cta_amax`: `live_localcta_cta_amax_experimental/localcta` MAE `0.001651380`, norm-MAE `0.159753868`, RMSE `0.002846398`, norm-RMSE `0.275359428`
- `student_t5` / `S=4096`: winner `localcta_cta_amax`
- `mxfp4_online`: `live_direct/mxfp4_v3` MAE `0.001433530`, norm-MAE `0.196442886`, RMSE `0.002489432`, norm-RMSE `0.341137774`
- `sa3_baseline`: `live_sa3_baseline/localcta` MAE `0.001197070`, norm-MAE `0.164039766`, RMSE `0.002121629`, norm-RMSE `0.290736189`
- `localcta_cta_amax`: `live_localcta_cta_amax_experimental/localcta` MAE `0.001178229`, norm-MAE `0.161457869`, RMSE `0.002106840`, norm-RMSE `0.288709463`
- `student_t5` / `S=8192`: winner `localcta_cta_amax`
- `mxfp4_online`: `live_direct/mxfp4_v3` MAE `0.001017332`, norm-MAE `0.195979689`, RMSE `0.001844229`, norm-RMSE `0.355273713`
- `sa3_baseline`: `live_sa3_baseline/localcta` MAE `0.000852565`, norm-MAE `0.164238777`, RMSE `0.001568679`, norm-RMSE `0.302191543`
- `localcta_cta_amax`: `live_localcta_cta_amax_experimental/localcta` MAE `0.000839629`, norm-MAE `0.161746751`, RMSE `0.001557629`, norm-RMSE `0.300062829`
- `signed_lognormal` / `S=2048`: winner `localcta_cta_amax`
- `mxfp4_online`: `live_direct/mxfp4_v3` MAE `0.002093406`, norm-MAE `0.204901202`, RMSE `0.003623498`, norm-RMSE `0.354665660`
- `sa3_baseline`: `live_sa3_baseline/localcta` MAE `0.001658221`, norm-MAE `0.162305639`, RMSE `0.002837477`, norm-RMSE `0.277730449`
- `localcta_cta_amax`: `live_localcta_cta_amax_experimental/localcta` MAE `0.001630653`, norm-MAE `0.159607282`, RMSE `0.002812517`, norm-RMSE `0.275287402`
- `signed_lognormal` / `S=4096`: winner `localcta_cta_amax`
- `mxfp4_online`: `live_direct/mxfp4_v3` MAE `0.001512593`, norm-MAE `0.200987452`, RMSE `0.002668959`, norm-RMSE `0.354640882`
- `sa3_baseline`: `live_sa3_baseline/localcta` MAE `0.001194407`, norm-MAE `0.158708168`, RMSE `0.002121165`, norm-RMSE `0.281852158`
- `localcta_cta_amax`: `live_localcta_cta_amax_experimental/localcta` MAE `0.001172340`, norm-MAE `0.155775993`, RMSE `0.002101107`, norm-RMSE `0.279186901`
- `signed_lognormal` / `S=8192`: winner `localcta_cta_amax`
- `mxfp4_online`: `live_direct/mxfp4_v3` MAE `0.001068621`, norm-MAE `0.208315110`, RMSE `0.002018435`, norm-RMSE `0.393470203`
- `sa3_baseline`: `live_sa3_baseline/localcta` MAE `0.000862382`, norm-MAE `0.168111354`, RMSE `0.001610767`, norm-RMSE `0.314000107`
- `localcta_cta_amax`: `live_localcta_cta_amax_experimental/localcta` MAE `0.000846804`, norm-MAE `0.165074583`, RMSE `0.001597306`, norm-RMSE `0.311376107`
- `gaussian_spikes` / `S=2048`: winner `localcta_cta_amax`
- `mxfp4_online`: `live_direct/mxfp4_v3` MAE `0.002121420`, norm-MAE `0.215143123`, RMSE `0.003635517`, norm-RMSE `0.368694767`
- `sa3_baseline`: `live_sa3_baseline/localcta` MAE `0.001605381`, norm-MAE `0.162809165`, RMSE `0.002838384`, norm-RMSE `0.287853754`
- `localcta_cta_amax`: `live_localcta_cta_amax_experimental/localcta` MAE `0.001583319`, norm-MAE `0.160571768`, RMSE `0.002827622`, norm-RMSE `0.286762338`
- `gaussian_spikes` / `S=4096`: winner `localcta_cta_amax`
- `mxfp4_online`: `live_direct/mxfp4_v3` MAE `0.001586003`, norm-MAE `0.217542512`, RMSE `0.002798093`, norm-RMSE `0.383797747`
- `sa3_baseline`: `live_sa3_baseline/localcta` MAE `0.001201727`, norm-MAE `0.164833679`, RMSE `0.002135465`, norm-RMSE `0.292908989`
- `localcta_cta_amax`: `live_localcta_cta_amax_experimental/localcta` MAE `0.001184573`, norm-MAE `0.162480827`, RMSE `0.002119989`, norm-RMSE `0.290786179`
- `gaussian_spikes` / `S=8192`: winner `localcta_cta_amax`
- `mxfp4_online`: `live_direct/mxfp4_v3` MAE `0.001140224`, norm-MAE `0.224282629`, RMSE `0.002156556`, norm-RMSE `0.424195602`
- `sa3_baseline`: `live_sa3_baseline/localcta` MAE `0.000850930`, norm-MAE `0.167378358`, RMSE `0.001606632`, norm-RMSE `0.316025186`
- `localcta_cta_amax`: `live_localcta_cta_amax_experimental/localcta` MAE `0.000839119`, norm-MAE `0.165055043`, RMSE `0.001598707`, norm-RMSE `0.314466399`
