# Changelog

## [1.1.3](https://github.com/AlexsJones/llmfit/compare/v1.1.2...v1.1.3) (2026-07-14)


### Features

* add NCAI VAETKI model to curated model database ([bdbf338](https://github.com/AlexsJones/llmfit/commit/bdbf338010ebeac9d5797beb42993b8c31845bcd))
* add RamaLama as a detection provider ([5bef0b2](https://github.com/AlexsJones/llmfit/commit/5bef0b2e49920acee70a354e802f8656097106fb))


### Bug Fixes

* **hardware:** read discrete Intel Arc VRAM from sysfs (xe/i915) ([9bbaa8d](https://github.com/AlexsJones/llmfit/commit/9bbaa8d0542ab2e5ad4ee5691e95b6ddf37667d6))
* **hardware:** read discrete Intel Arc VRAM from sysfs (xe/i915) ([9fbbcb8](https://github.com/AlexsJones/llmfit/commit/9fbbcb8312e6989d7b35f7ae24c7b36562ccce68)), closes [#609](https://github.com/AlexsJones/llmfit/issues/609)
* match Unsloth `-UD` dynamic GGUFs to catalog models ([b3de39d](https://github.com/AlexsJones/llmfit/commit/b3de39dac4339df45f4ddff9ac71a6dbbf9fdc10))
* match Unsloth `-UD` dynamic GGUFs to catalog models ([e9b66be](https://github.com/AlexsJones/llmfit/commit/e9b66be1506095fbd905fbe3d9b9c2d1c13a7f03))
* name AMD unified APUs when rocm-smi reports "N/A", gate junk shares ([ba96f4c](https://github.com/AlexsJones/llmfit/commit/ba96f4c3cbdb6e1cc2dcbb8db7f390e625b02193))
* name AMD unified APUs when rocm-smi reports "N/A", gate junk shares ([e282011](https://github.com/AlexsJones/llmfit/commit/e282011f58bf0582ad606a907d51c9b743ea0b6e))
* **providers:** detect oMLX as an MLX runtime ([f649952](https://github.com/AlexsJones/llmfit/commit/f649952c39028dd2da06c3461efe8b43b9fdbe3b))
* **providers:** show LM Studio/Docker Desktop as installed when idle ([bca491e](https://github.com/AlexsJones/llmfit/commit/bca491e380fbeb37719a80b6f2f94838750078b9))
* **providers:** show LM Studio/Docker Desktop as installed when idle ([5a22add](https://github.com/AlexsJones/llmfit/commit/5a22add2266e9079e6853f187de1fff3904ff6b1)), closes [#731](https://github.com/AlexsJones/llmfit/issues/731)
* **serve:** include per-GPU memory_bandwidth_gbps in /api/v1/system ([ce3f7bf](https://github.com/AlexsJones/llmfit/commit/ce3f7bfcccd56567a1c5b75c5770347d1acd668b))
* **serve:** include per-GPU memory_bandwidth_gbps in /api/v1/system ([8b9b7aa](https://github.com/AlexsJones/llmfit/commit/8b9b7aa3c2cbf05028cfbd44a636d512bb87e687)), closes [#747](https://github.com/AlexsJones/llmfit/issues/747)

## [1.1.2](https://github.com/AlexsJones/llmfit/compare/v1.1.1...v1.1.2) (2026-07-10)


### Features

* **share:** ship the registered OAuth App client id — interactive login enabled ([a94ffcb](https://github.com/AlexsJones/llmfit/commit/a94ffcbe9412f0974224c87290ae01f89d01e009))
* **share:** ship the registered OAuth App client id — interactive login enabled ([325f8a3](https://github.com/AlexsJones/llmfit/commit/325f8a3ff34adf0772990335e93f1efad212d735))

## [1.1.1](https://github.com/AlexsJones/llmfit/compare/v1.1.0...v1.1.1) (2026-07-10)


### Features

* **bench:** ship merged community benchmarks to every user in the next release ([f2fd476](https://github.com/AlexsJones/llmfit/commit/f2fd47672ff1b81bacdba3169e163f235a32070a))
* **bench:** ship merged community benchmarks to every user in the next release ([5c46f92](https://github.com/AlexsJones/llmfit/commit/5c46f92a2411edbfa7a829cc0e4c92f355189d76))

## [1.1.0](https://github.com/AlexsJones/llmfit/compare/v1.0.1...v1.1.0) (2026-07-10)


### Features

* bench --share — contribute benchmarks as PRs without the gh CLI (RFC [#710](https://github.com/AlexsJones/llmfit/issues/710)) ([5114694](https://github.com/AlexsJones/llmfit/commit/51146943d75d09faeb7cc7412f32984638679ee7))
* **bench:** add --share to contribute benchmarks via PR without gh CLI ([721eb66](https://github.com/AlexsJones/llmfit/commit/721eb666ab4c0fc15e3ea75a510aacb127d04a5d))
* **bench:** detect llama-server as a first-class provider ([6dc367b](https://github.com/AlexsJones/llmfit/commit/6dc367be97329be00301a8aeeb63fc753d0c9ac2))
* **bench:** local benchmark store — save runs, share the backlog later ([0e22a75](https://github.com/AlexsJones/llmfit/commit/0e22a75abc2da4da0f8c51797f79f391fc74cd3c))
* **bench:** local store follow-ups — share backlog, dedup PRs, local-measurement overrides + calibration ([390cbf1](https://github.com/AlexsJones/llmfit/commit/390cbf1c6726580c308cc9113df0eede7ae7dba1))
* **fit:** local benchmark runs override estimated tok/s ([274c048](https://github.com/AlexsJones/llmfit/commit/274c048bc3366bc74851128062cce27e81287d88))
* **fit:** purge test-stub catalog entries; calibrate estimates from local benchmarks ([fb049c2](https://github.com/AlexsJones/llmfit/commit/fb049c210c13f65c0d77ac8da0ce50aec7c0b332))
* **share:** reuse open benchmark PR and make submissions idempotent ([52f3afb](https://github.com/AlexsJones/llmfit/commit/52f3afb10b6fb521d880bae896cae15b117d9c69))
* **tui:** expose installed/GGUF availability filter in the Filter modal ([dd940d5](https://github.com/AlexsJones/llmfit/commit/dd940d5288980c8ee9cac5724d4a75318d4be9d3))
* **tui:** filter by installed models in the Filter modal ([75f4f1e](https://github.com/AlexsJones/llmfit/commit/75f4f1ec773435288b59ff83e699091bef253820))
* **tui:** offer to benchmark the selected model on entering leaderboard ([61f6a47](https://github.com/AlexsJones/llmfit/commit/61f6a4705c5b72e0838aa957ef4507efde86bfaa))


### Bug Fixes

* **data:** verify GGUF source provenance; unlink wrong-model quants ([63b2d3a](https://github.com/AlexsJones/llmfit/commit/63b2d3aada13d87b276666497074c8c816d8805b))
* **providers:** stop marking whole model families as installed for llama.cpp ([f839633](https://github.com/AlexsJones/llmfit/commit/f8396335b0fe748555e0764f0386ab7e17fac73e))
* **tests:** pin GitHub-credential state in bench-offer toggle test ([618ba16](https://github.com/AlexsJones/llmfit/commit/618ba162a146b623711202264a3405695be1a037))
* **tests:** pin GitHub-credential state in bench-offer toggle test ([fb5caff](https://github.com/AlexsJones/llmfit/commit/fb5caffc371907d53578c1919bf6ffa5a1616b11))

## [1.0.1](https://github.com/AlexsJones/llmfit/compare/v1.0.0...v1.0.1) (2026-07-09)


### Bug Fixes

* bump internal llmfit-core version requirement to 1.0.0 ([7274a45](https://github.com/AlexsJones/llmfit/commit/7274a450a04107b3272cf958a5f20e18a516a406))

## [1.0.0](https://github.com/AlexsJones/llmfit/compare/v0.9.38...v1.0.0) (2026-07-09)


### Features

* add global --tui flag to force interactive TUI ([749b22f](https://github.com/AlexsJones/llmfit/commit/749b22fcd30d72fb2c57a1b1c91af3d834c68c3f))
* add global --tui flag to force interactive TUI ([#547](https://github.com/AlexsJones/llmfit/issues/547)) ([f0146b1](https://github.com/AlexsJones/llmfit/commit/f0146b1b4e3d1233fe9a3dc4dcbcc84f5aa6ca83))
* add model language metadata ([784af4d](https://github.com/AlexsJones/llmfit/commit/784af4d8c17c6c16762c0ca9fb1f0a19072916a3))
* add TTS model discovery ([40a69f6](https://github.com/AlexsJones/llmfit/commit/40a69f62697a5d92af38f8f24d337b3b2eac648f))
* auto-detect system RAM bandwidth for MoE-offload estimates ([c10f567](https://github.com/AlexsJones/llmfit/commit/c10f567a7c355e5d52586a44d778b7553fed2bfd))
* auto-detect system RAM bandwidth for MoE-offload estimates ([f469543](https://github.com/AlexsJones/llmfit/commit/f469543c9c13b762748af20c60ec9fb9ea971c3d))
* discover TTS language metadata in scraper ([74cd130](https://github.com/AlexsJones/llmfit/commit/74cd1306197000a8573cb102a51e4ca47262d923))
* estimate basis + measured tok/s provenance ([#292](https://github.com/AlexsJones/llmfit/issues/292)) ([93f6806](https://github.com/AlexsJones/llmfit/commit/93f6806fb0cd50fec555ab0166acc71f1d483231))
* expose estimate basis and local verification commands ([#292](https://github.com/AlexsJones/llmfit/issues/292)) ([af54caf](https://github.com/AlexsJones/llmfit/commit/af54caf3a35c85f394df2c22ba26c9c6b6b5a98c))
* pin Huihui-Qwen3.6-35B-A3B-abliterated and add GGUF source ([d048900](https://github.com/AlexsJones/llmfit/commit/d048900953dc4196a5bf59a4af692558cb5cc282))
* pin Huihui-Qwen3.6-35B-A3B-abliterated and add GGUF source ([#656](https://github.com/AlexsJones/llmfit/issues/656)) ([e0a5654](https://github.com/AlexsJones/llmfit/commit/e0a56545387b05f4f966a0e8e5d07c0c66a14276))
* surface community-measured tok/s when hardware matches benchmark data ([34eca7e](https://github.com/AlexsJones/llmfit/commit/34eca7e8432c42ccf0fb20a36b1e839f827dfb80))
* surface community-measured tok/s when hardware matches benchmark data ([730cc5d](https://github.com/AlexsJones/llmfit/commit/730cc5d1e404a672671c9d2633866d1557c38002))


### Bug Fixes

* harden AMD sysfs GPU detection against silent card drops ([#303](https://github.com/AlexsJones/llmfit/issues/303), [#638](https://github.com/AlexsJones/llmfit/issues/638)) ([84a6cfa](https://github.com/AlexsJones/llmfit/commit/84a6cfa7c430906d77b8803fec28e5c362461a3c))
* harden AMD sysfs GPU detection against silent card drops ([#303](https://github.com/AlexsJones/llmfit/issues/303), [#638](https://github.com/AlexsJones/llmfit/issues/638)) ([4dc94ca](https://github.com/AlexsJones/llmfit/commit/4dc94ca3fca3601dcbf040128ae4ed9b699875e3))
* keep text generation discovery budget ([5b55e2c](https://github.com/AlexsJones/llmfit/commit/5b55e2c5710f28b2ff79e9cde4ba4f94df6f9fd2))
* mark TTS models as unsupported runtime ([5ca524e](https://github.com/AlexsJones/llmfit/commit/5ca524e6ea8a23d93c4a130971f99f837c82226d))
* **ollama:** don't mark cloud models as locally installed (closes [#619](https://github.com/AlexsJones/llmfit/issues/619)) ([580afee](https://github.com/AlexsJones/llmfit/commit/580afee47da41837d27f51cadfba99e1cbed4022))
* preserve popularity metadata through scraper merge and discovery ([b2b35d6](https://github.com/AlexsJones/llmfit/commit/b2b35d60d15e93953e73061a336ad0368c04572c))
* resolve [#171](https://github.com/AlexsJones/llmfit/issues/171) — Include ONNX models. ([9939e67](https://github.com/AlexsJones/llmfit/commit/9939e67b309b24252e1838bdc231d3a1d0885704))
* support TTS catalog metadata ([c98b3d5](https://github.com/AlexsJones/llmfit/commit/c98b3d5fa72418b65a1e84c5cb5d1eb8a310b857))
* use cross-platform which crate for binary detection in TUI ([55d77ec](https://github.com/AlexsJones/llmfit/commit/55d77ecd0f66f670641ead6da19ffd8fec2bcf3c))
* use cross-platform which crate for binary detection in TUI ([d450daa](https://github.com/AlexsJones/llmfit/commit/d450daac08313145687d366d1cc6dc4db57046fe))
* wire onnx catalog into model database ([f2e7dba](https://github.com/AlexsJones/llmfit/commit/f2e7dbafce1aba08e76d18494c1e556141ac2a13))


### Miscellaneous Chores

* cut 1.0.0 ([21ebc84](https://github.com/AlexsJones/llmfit/commit/21ebc847dede0eb05b214296c075d8d2f21c51c5))

## [0.9.38](https://github.com/AlexsJones/llmfit/compare/v0.9.37...v0.9.38) (2026-07-04)


### Bug Fixes

* **docker:** drop COPY of removed root data/ directory ([a4445c7](https://github.com/AlexsJones/llmfit/commit/a4445c70885ee399a5b8e39b4ded938a09230064))
* **docker:** drop COPY of removed root data/ directory ([5bb514d](https://github.com/AlexsJones/llmfit/commit/5bb514d130964311617269b0cffd4663a31522e3))

## [0.9.37](https://github.com/AlexsJones/llmfit/compare/v0.9.36...v0.9.37) (2026-07-03)


### Features

* **calibration:** weekly benchmark refresh + estimate-accuracy audit ([#112](https://github.com/AlexsJones/llmfit/issues/112)/[#119](https://github.com/AlexsJones/llmfit/issues/119)) ([5548723](https://github.com/AlexsJones/llmfit/commit/55487236685fde60a2be09580fe892daeb6a9b8b))
* **calibration:** weekly benchmark refresh + estimate-accuracy audit against measured data ([6c40dcf](https://github.com/AlexsJones/llmfit/commit/6c40dcf3127ada85a47cb4023c41fee28d33ec3e))
* **claim:** JSON output for programmatic consumers (llmfit-dra M0) ([f469c22](https://github.com/AlexsJones/llmfit/commit/f469c22cb48cc8e15b33119cd7cf166474ab50a7))
* **claim:** machine-readable JSON output via the global --json flag ([1636e51](https://github.com/AlexsJones/llmfit/commit/1636e51bb8b452b6c9ab8e47f38e0ac764f25f2f))


### Bug Fixes

* **calibration:** exclude speculative-decoding/MTP runs from the accuracy audit ([1c058aa](https://github.com/AlexsJones/llmfit/commit/1c058aa22bd57456487a15aa618155fd1340ef26))

## [0.9.36](https://github.com/AlexsJones/llmfit/compare/v0.9.35...v0.9.36) (2026-07-03)


### Features

* add 'llmfit doctor' hardware diagnostic report for bug reports ([53663bc](https://github.com/AlexsJones/llmfit/commit/53663bcb9935d79c6f959b357d99d622d6cbcef1))
* add `llmfit doctor` hardware diagnostic report ([bcd7dc0](https://github.com/AlexsJones/llmfit/commit/bcd7dc0c89b48909cfdd0ce20813bd58a2e84aeb))
* add issue templates for bug reports ([bab1fe9](https://github.com/AlexsJones/llmfit/commit/bab1fe9166a037aac1e3d4da10484e0f91812bc1))
* request 'llmfit doctor' output in the bug report template ([b4069d9](https://github.com/AlexsJones/llmfit/commit/b4069d927b9f7d271664d9e24324b3fe5716fa29))
* **scoring:** benchmark-informed task alignment per use case ([ce6d37e](https://github.com/AlexsJones/llmfit/commit/ce6d37e034b3978e687a00589ac7ce10e81ae7e9))
* **scoring:** benchmark-informed task alignment per use case (issue [#150](https://github.com/AlexsJones/llmfit/issues/150)) ([c994615](https://github.com/AlexsJones/llmfit/commit/c994615357f1b2577ba5096d7d176cf928f27a56))
* serve over a Unix domain socket (--unix-socket) ([de62c2b](https://github.com/AlexsJones/llmfit/commit/de62c2bc7e2af432771e0036b35e5fb76a50ed7e))
* serve over a Unix domain socket (--unix-socket) ([03d6ca5](https://github.com/AlexsJones/llmfit/commit/03d6ca5170df4f924b3bf24e94f27035fa26bb6a))
* show usable context per model (native→achievable) in TUI, CLI, and JSON ([e33a3a3](https://github.com/AlexsJones/llmfit/commit/e33a3a31be5a7a7b3953c4f33a02732c1790353c)), closes [#621](https://github.com/AlexsJones/llmfit/issues/621)
* usable-context display — native→achievable (closes [#621](https://github.com/AlexsJones/llmfit/issues/621)) ([1706929](https://github.com/AlexsJones/llmfit/commit/1706929ea367fd137117173f56827640bc04404c))
* user-local custom model overlay (closes [#451](https://github.com/AlexsJones/llmfit/issues/451)) ([decef82](https://github.com/AlexsJones/llmfit/commit/decef827b3785df9a5c785b49ed128f98d5c3aaa))
* user-local custom model overlay (custom_models.json) ([e55c8de](https://github.com/AlexsJones/llmfit/commit/e55c8dec388a1788b6093e0ef4c1091149cfe3d9)), closes [#451](https://github.com/AlexsJones/llmfit/issues/451)


### Bug Fixes

* guard optional attribute lookups in generated claim CEL ([c6f481f](https://github.com/AlexsJones/llmfit/commit/c6f481f27557d3e0fd6fc584fd1b1fa008ff95a4))
* guard optional attribute lookups in generated claim CEL ([a6f3c6a](https://github.com/AlexsJones/llmfit/commit/a6f3c6a6bc11e3a4c141dc7e0c3891263958a712))
* **hardware:** correct Intel GPU detection — iGPUs get the unified RAM pool, dGPUs get Vulkan VRAM ([4834767](https://github.com/AlexsJones/llmfit/commit/4834767462a7ac939ddb0306796e9661913f8310))
* **hardware:** detect generic-named MI50-class accelerators (issue [#638](https://github.com/AlexsJones/llmfit/issues/638)) ([180f484](https://github.com/AlexsJones/llmfit/commit/180f4844630e4ca08e8971482a615564d5be64f8))
* **hardware:** don't drop large-VRAM AMD accelerators with generic ROCm names ([bf16eb8](https://github.com/AlexsJones/llmfit/commit/bf16eb8e87b31e1c0f5c6154e82f7b8967d54085))
* **hardware:** Intel iGPU unified-memory pool + discrete Arc VRAM via Vulkan (issue [#609](https://github.com/AlexsJones/llmfit/issues/609)) ([a120aa3](https://github.com/AlexsJones/llmfit/commit/a120aa38d9410b2d874bbe7c70653b4f2c1920b7))
* **schema:** add 'audio' to capability enum ([a812232](https://github.com/AlexsJones/llmfit/commit/a812232f0eb02dc66544e508fe6e5e587acbd1f0))

## [0.9.35](https://github.com/AlexsJones/llmfit/compare/v0.9.34...v0.9.35) (2026-07-02)


### Features

* add claim subcommand generating Kubernetes DRA ResourceClaims ([08547ef](https://github.com/AlexsJones/llmfit/commit/08547efa277dd3f733ecdf72312a3376fbc2a01b))
* add claim subcommand generating Kubernetes DRA ResourceClaims ([3369a49](https://github.com/AlexsJones/llmfit/commit/3369a492630f357c65e2344de18b5a2e35ed4950))
* expose GPU memory bandwidth in system JSON and add Strix Halo entries ([3bfd334](https://github.com/AlexsJones/llmfit/commit/3bfd33406b071f48fddc7fb6e99fe62e55096511))


### Bug Fixes

* reset list cursor to top when the search query changes ([5e4dc7b](https://github.com/AlexsJones/llmfit/commit/5e4dc7b1e578c5a2366a8f7b94220d68178e2fae))
* reset list cursor to top when the search query changes ([43884f5](https://github.com/AlexsJones/llmfit/commit/43884f500a9b392b03bc2d2bc539ba6430d954fd)), closes [#657](https://github.com/AlexsJones/llmfit/issues/657)

## [0.9.34](https://github.com/AlexsJones/llmfit/compare/v0.9.33...v0.9.34) (2026-06-28)


### Bug Fixes

* pin Docker builder to bookworm to match runtime glibc ([f13a633](https://github.com/AlexsJones/llmfit/commit/f13a6332e62713b22bf96b7a26b551f43caed831))
* pin Docker builder to bookworm to match runtime glibc ([48caf17](https://github.com/AlexsJones/llmfit/commit/48caf178baaf7e7c0f81bfd30f4972b404528cf7)), closes [#655](https://github.com/AlexsJones/llmfit/issues/655)

## [0.9.33](https://github.com/AlexsJones/llmfit/compare/v0.9.32...v0.9.33) (2026-06-23)


### Bug Fixes

* **docker:** bump builder image to rust 1.95 (sysinfo MSRV) ([b63a142](https://github.com/AlexsJones/llmfit/commit/b63a14244d93fb1fdd50b2f49c96abaac872b977))
* **docker:** bump builder image to rust 1.95 (sysinfo MSRV) ([b943f3a](https://github.com/AlexsJones/llmfit/commit/b943f3a84a63e77a740acacd38a09c69774dda8c))

## [0.9.32](https://github.com/AlexsJones/llmfit/compare/v0.9.31...v0.9.32) (2026-06-23)


### Features

* add NCAI VAETKI models to curated model database ([b3c04a8](https://github.com/AlexsJones/llmfit/commit/b3c04a82055875b27a8808131df8f41e544ca759))
* **analysis:** add InstalledIndex and build_model_fits to consolidate provider detection ([33dce8b](https://github.com/AlexsJones/llmfit/commit/33dce8bbb9f8a527c48fec07152fa2c3a0983784))
* **data:** add Whisper/ASR model entries + audio support roadmap ([27bae48](https://github.com/AlexsJones/llmfit/commit/27bae4878f798c7069f936de6fa9ed5b1dcc3312))
* **python:** add riscv64 wheel build ([e12bb85](https://github.com/AlexsJones/llmfit/commit/e12bb85abdd752e632930c3086a4afc935e9a4a6))
* **python:** add riscv64 wheel build ([cb377f4](https://github.com/AlexsJones/llmfit/commit/cb377f45595b745d7bf0963ab1c86b807c210b01))


### Bug Fixes

* bundle WebView2 bootstrapper for Windows desktop app ([c72103b](https://github.com/AlexsJones/llmfit/commit/c72103b784134b8cc66725d4e8e9c36b0d01e64d))
* **fit:** CPU-only models that fit with headroom reach Good, not Marginal ([aad4c37](https://github.com/AlexsJones/llmfit/commit/aad4c373285e328bcbab72577ee49d62cca1515d))
* **fit:** CPU-only models that fit with headroom reach Good, not Marginal ([0ced1b7](https://github.com/AlexsJones/llmfit/commit/0ced1b775cb39f783cabf4411d25e1e1aa188015))
* **hardware:** detect all GPUs across sysfs, ROCm tabular, and fit pool ([0989ca2](https://github.com/AlexsJones/llmfit/commit/0989ca2dc443f782cf34cec709e989c23f234178))
* **hardware:** ROCm tabular multi-GPU parsing + summed fit VRAM pool ([4e27617](https://github.com/AlexsJones/llmfit/commit/4e27617baaeaf1902131f59c71763de360ef6837))
* **scoring:** MoE active-param quality, recency bonus, smooth fit curve ([c499c00](https://github.com/AlexsJones/llmfit/commit/c499c000948cb620564d9c8433df5b8b5d36880d))
* **scoring:** MoE active-param quality, recency bonus, smooth fit curve ([2db3e55](https://github.com/AlexsJones/llmfit/commit/2db3e552ad637437589e692bf91bc9ca69350925))
* **scoring:** treat headroom as a perfect fit (one-sided fit curve) ([235be33](https://github.com/AlexsJones/llmfit/commit/235be33b1d8ada30693f72b48b48d50bd48df5a9))
* **tui:** make provider filter input cursor-aware and ASCII-only ([933644e](https://github.com/AlexsJones/llmfit/commit/933644e4d1a80969cb7bfc1c5386423291a50132))


### Performance Improvements

* **analysis:** parallelize provider detection in detect_all with thread::scope ([d0660bc](https://github.com/AlexsJones/llmfit/commit/d0660bc72e601a7707caabb2f4e042a0701c491e))

## [0.9.31](https://github.com/AlexsJones/llmfit/compare/v0.9.30...v0.9.31) (2026-06-09)


### Features

* **lmstudio:** Add LMSTUDIO_API_KEY support for authenticated instances ([0dc2712](https://github.com/AlexsJones/llmfit/commit/0dc2712df4b4d947e0db371c98bf376d4da16c85))
* **lmstudio:** Add LMSTUDIO_API_KEY support for authenticated instances ([302cb53](https://github.com/AlexsJones/llmfit/commit/302cb535830e7e001c0735596d0c06da96fdc467))
* speed up provider navigation when holding shift (gh-571) ([f0d4783](https://github.com/AlexsJones/llmfit/commit/f0d4783dd99f4c12a0fe15d564d8bc787ef7c755))
* speed up provider navigation when holding shift (gh-571) ([7d74d7d](https://github.com/AlexsJones/llmfit/commit/7d74d7d4a93cb8c229868b00d8bc3385005e3551))
* upgrade MiniMax default model to M3 ([d45a546](https://github.com/AlexsJones/llmfit/commit/d45a546ebd9eca78cb5e09f191321e67807112d2))
* upgrade MiniMax default model to M3 ([7407d10](https://github.com/AlexsJones/llmfit/commit/7407d1035ef7dd7b14ed50a1dc1f3f6e21a08a44))


### Bug Fixes

* add Gemma 4 QAT models and fallback HF search ([e40d3b8](https://github.com/AlexsJones/llmfit/commit/e40d3b8e00b1db1fae9234e34fadde8a8a3945a2))
* add Gemma 4 QAT models and fallback HF search ([9b4582f](https://github.com/AlexsJones/llmfit/commit/9b4582f0e45c02ffb09abaef649cc45e01344f63)), closes [#613](https://github.com/AlexsJones/llmfit/issues/613)
* **lmstudio:** empty string guard, README docs, and unit test for LMSTUDIO_API_KEY ([46f70b7](https://github.com/AlexsJones/llmfit/commit/46f70b790e3bd177e001469d04456ce1e20193c5))
* replace unsafe env mutation in tests with pure logic tests ([33047d8](https://github.com/AlexsJones/llmfit/commit/33047d85fab2913d09905922df9a3daef6b5a314))
* **tui:** handle search input cursor and overflow ([66f4c88](https://github.com/AlexsJones/llmfit/commit/66f4c887b37c617944a8bc14a1d0a20aedac9767))
* **tui:** handle search input cursor and overflow ([8596005](https://github.com/AlexsJones/llmfit/commit/85960054bbb88c961ffa4797006dd3475a8b4699))
* **tui:** handle unicode download directory input ([b34d164](https://github.com/AlexsJones/llmfit/commit/b34d164cb9f5baf60f3b14058599d0d13755e212))
* **tui:** handle unicode download directory input ([da120f8](https://github.com/AlexsJones/llmfit/commit/da120f8d6993dd2e6cfed899478b6796b1992221))
* **tui:** ignore modified key input in search ([6c0b697](https://github.com/AlexsJones/llmfit/commit/6c0b69701e8ec9be28a7a98bd0e94812f64a037c))
* **tui:** ignore modified key input in search ([8de0a1b](https://github.com/AlexsJones/llmfit/commit/8de0a1b5a0627327f82360f9c9ced2ca71cfc4a7))
* use GGUF metadata for param counts in GGUF-only repos ([3f9b22b](https://github.com/AlexsJones/llmfit/commit/3f9b22bb032a3b4444750ddfca12acfdf96e27df))
* use GGUF metadata for parameter counts in GGUF-only repos ([ff28cdd](https://github.com/AlexsJones/llmfit/commit/ff28cdd002f4777ab1857b0c6e669c3c7789a1cd)), closes [#622](https://github.com/AlexsJones/llmfit/issues/622)
* wrap env::set_var/remove_var in unsafe blocks for Rust 2024 edition ([55b7979](https://github.com/AlexsJones/llmfit/commit/55b79793d1aa8bd3ca6e6ab7d4f11c5f20f897f6))

## [0.9.30](https://github.com/AlexsJones/llmfit/compare/v0.9.29...v0.9.30) (2026-06-01)


### Features

* **tui:** fuzzy search in provider filter popup ([#600](https://github.com/AlexsJones/llmfit/issues/600)) ([b055a04](https://github.com/AlexsJones/llmfit/commit/b055a04beb50008c3d868aa7f43598ad6b614d33))

## [0.9.29](https://github.com/AlexsJones/llmfit/compare/v0.9.28...v0.9.29) (2026-05-27)


### Features

* open TUI with best-fit model preselected ([#582](https://github.com/AlexsJones/llmfit/issues/582)) ([834ee20](https://github.com/AlexsJones/llmfit/commit/834ee20ca2d3cb18fa10292afce6f9ba75acb9e5))
* rescrape model catalog and expand Ollama mappings ([#573](https://github.com/AlexsJones/llmfit/issues/573)) ([c3e3f0d](https://github.com/AlexsJones/llmfit/commit/c3e3f0d03fb9a83ced7a4c4ff0ba2ad6926db153))


### Bug Fixes

* align SortColumn::next with display order of columns in TUI ([#576](https://github.com/AlexsJones/llmfit/issues/576)) ([9e3477e](https://github.com/AlexsJones/llmfit/commit/9e3477e2165f0b379a003c64a9651fcfd54de6db))
* **core:** scan XDG cache path for HF models on macOS ([#568](https://github.com/AlexsJones/llmfit/issues/568)) ([#575](https://github.com/AlexsJones/llmfit/issues/575)) ([9ec7dd3](https://github.com/AlexsJones/llmfit/commit/9ec7dd35f7ab603e16caee163ed396f02a30a3ee))
* **release:** unblock downstream publish jobs when sign-windows fails ([18b581f](https://github.com/AlexsJones/llmfit/commit/18b581fcf2fd26b50bd0b559a71c2adc86f953f3)), closes [#581](https://github.com/AlexsJones/llmfit/issues/581)
* **release:** use always() to actually override transitive skip ([0d6aeb3](https://github.com/AlexsJones/llmfit/commit/0d6aeb3e245ec0574929717c8954f74a26120996))

## [0.9.28](https://github.com/AlexsJones/llmfit/compare/v0.9.27...v0.9.28) (2026-05-22)


### Features

* **tui:** add ollama_name field to JSON output ([#566](https://github.com/AlexsJones/llmfit/issues/566)) ([36b3181](https://github.com/AlexsJones/llmfit/commit/36b3181ccd8d6343d06a0de05f3a3838cc055e50))


### Bug Fixes

* **core:** detect GGUF models in HuggingFace cache for llama.cpp ([#568](https://github.com/AlexsJones/llmfit/issues/568)) ([6093bcc](https://github.com/AlexsJones/llmfit/commit/6093bcc952eee0b5a64141809665c8bc823e21d8))
* **tui:** include GGUF source providers in search and provider filter ([#569](https://github.com/AlexsJones/llmfit/issues/569)) ([8698b99](https://github.com/AlexsJones/llmfit/commit/8698b990aa73622e124c87ad2257fa32cae87e34))

## [0.9.27](https://github.com/AlexsJones/llmfit/compare/v0.9.26...v0.9.27) (2026-05-20)


### Bug Fixes

* **tui:** correct live-bench keybinding label from B to I ([7aae532](https://github.com/AlexsJones/llmfit/commit/7aae5326ef15018fccb9d1d9605252f75739a749))

## [0.9.26](https://github.com/AlexsJones/llmfit/compare/v0.9.25...v0.9.26) (2026-05-19)


### Bug Fixes

* **hardware:** detect distinct AMD GPU models in multi-GPU ROCm systems ([#559](https://github.com/AlexsJones/llmfit/issues/559)) ([a100552](https://github.com/AlexsJones/llmfit/commit/a100552a60876ad84ee3b092704a1f4b4dadc143)), closes [#550](https://github.com/AlexsJones/llmfit/issues/550)

## [0.9.25](https://github.com/AlexsJones/llmfit/compare/v0.9.24...v0.9.25) (2026-05-17)


### Features

* **scoring:** add generation-aware quality scoring ([#553](https://github.com/AlexsJones/llmfit/issues/553)) ([a3e0eb2](https://github.com/AlexsJones/llmfit/commit/a3e0eb27c57fe05298292f4c61699b9aed2824ea))
* **tui:** move provider detection to background threads for faster startup ([#554](https://github.com/AlexsJones/llmfit/issues/554)) ([9e8308d](https://github.com/AlexsJones/llmfit/commit/9e8308d3e6dccf83da35694233a53261902ebbf2))


### Bug Fixes

* **docker:** skip HTTP probe on Linux when Docker Desktop is not running ([#548](https://github.com/AlexsJones/llmfit/issues/548)) ([ce735c9](https://github.com/AlexsJones/llmfit/commit/ce735c95c39ab10bd936238b7ab611d336a13c12))

## [0.9.24](https://github.com/AlexsJones/llmfit/compare/v0.9.23...v0.9.24) (2026-05-12)


### Features

* add MCP server mode and NATS event publishing to llmfit serve ([#545](https://github.com/AlexsJones/llmfit/issues/545)) ([c4958f7](https://github.com/AlexsJones/llmfit/commit/c4958f7edb59225ffd569779fd8442915720bf03))


### Bug Fixes

* **hardware:** fix iGPU filtering on Ryzen 9 9950X where 2 GB iGPU passes the &gt;= 2 GB discrete GPU threshold ([#543](https://github.com/AlexsJones/llmfit/issues/543)) ([c655bc1](https://github.com/AlexsJones/llmfit/commit/c655bc1b2e10b5b0a13a123344755e450832a06d))
* MoE offload speed overestimate for Qwen3-Coder-Next ([#544](https://github.com/AlexsJones/llmfit/issues/544)) ([5bdc9df](https://github.com/AlexsJones/llmfit/commit/5bdc9df9d8e28168fdc4650741be71b715a11fc6))

## [0.9.23](https://github.com/AlexsJones/llmfit/compare/v0.9.22...v0.9.23) (2026-05-10)


### Features

* populate architecture metadata for precise KV cache and MoE speed estimation ([#539](https://github.com/AlexsJones/llmfit/issues/539)) ([bab8c01](https://github.com/AlexsJones/llmfit/commit/bab8c012771acec3bbd4a051fd3720379bb01e5d))


### Bug Fixes

* **bench:** implement 5 follow-up improvements to PR [#278](https://github.com/AlexsJones/llmfit/issues/278) ([#523](https://github.com/AlexsJones/llmfit/issues/523)) ([91fee77](https://github.com/AlexsJones/llmfit/commit/91fee7702a5728fd1b0b64459c8be024fc82051a))

## [0.9.22](https://github.com/AlexsJones/llmfit/compare/v0.9.21...v0.9.22) (2026-05-06)


### Features

* overhaul model discovery with cursor pagination, trending, and additive DB ([#537](https://github.com/AlexsJones/llmfit/issues/537)) ([93e9d67](https://github.com/AlexsJones/llmfit/commit/93e9d673cf54bde62dc7b24a1400c1cd89f70c1c))

## [0.9.21](https://github.com/AlexsJones/llmfit/compare/v0.9.20...v0.9.21) (2026-05-05)


### Bug Fixes

* add AutoRound format detection to prevent misclassification as GGUF ([#532](https://github.com/AlexsJones/llmfit/issues/532)) ([8214b1f](https://github.com/AlexsJones/llmfit/commit/8214b1f19bad2ed8e5b9e9ae98d203447c09fd36))

## [0.9.20](https://github.com/AlexsJones/llmfit/compare/v0.9.19...v0.9.20) (2026-05-04)


### Bug Fixes

* **lmstudio:** resolve HF repo IDs to direct GGUF file URLs for download ([#530](https://github.com/AlexsJones/llmfit/issues/530)) ([115a01d](https://github.com/AlexsJones/llmfit/commit/115a01d66e1a1689715dc1266d98a5887cc3d687))

## [0.9.19](https://github.com/AlexsJones/llmfit/compare/v0.9.18...v0.9.19) (2026-05-03)


### Features

* add llmfit bench for live inference benchmarking with quality metrics ([#278](https://github.com/AlexsJones/llmfit/issues/278)) ([8896fdb](https://github.com/AlexsJones/llmfit/commit/8896fdb861fa6950081c144ddf23d9eb24aef7ed))


### Bug Fixes

* **lmstudio:** wrap HuggingFace repo IDs in canonical URL for download ([#520](https://github.com/AlexsJones/llmfit/issues/520)) ([3399c70](https://github.com/AlexsJones/llmfit/commit/3399c70b6b2237c2285d92efe3ac1ebb2178b80e)), closes [#519](https://github.com/AlexsJones/llmfit/issues/519)
* **ollama:** treat OLLAMA_HOST=0.0.0.0 as a bind address and fall back to localhost ([#521](https://github.com/AlexsJones/llmfit/issues/521)) ([3a7bcf5](https://github.com/AlexsJones/llmfit/commit/3a7bcf52628161f142c9eed6151c9408f5af8ef4)), closes [#516](https://github.com/AlexsJones/llmfit/issues/516)

## [0.9.18](https://github.com/AlexsJones/llmfit/compare/v0.9.17...v0.9.18) (2026-04-30)


### Features

* **recommend:** add --output-llamacpp flag  ([#508](https://github.com/AlexsJones/llmfit/issues/508)) ([7c5925c](https://github.com/AlexsJones/llmfit/commit/7c5925c949878fe1015d815a1245734c75db1aca))


### Bug Fixes

* filter CPU Vulkan devices and suppress ROCm+Vulkan duplicates (fixes [#271](https://github.com/AlexsJones/llmfit/issues/271)) ([#328](https://github.com/AlexsJones/llmfit/issues/328)) ([fe8d3f0](https://github.com/AlexsJones/llmfit/commit/fe8d3f050a69b04589bf872e41001187fdd469ca))

## [0.9.17](https://github.com/AlexsJones/llmfit/compare/v0.9.16...v0.9.17) (2026-04-29)


### Features

* add benchmark cache with offline fallback ([928390b](https://github.com/AlexsJones/llmfit/commit/928390b467a78e9bf7b1f73f819acdea8044184b))
* add RWKV v7 G1f models (1.5B, 2.9B, 7.2B, 13.3B) ([#517](https://github.com/AlexsJones/llmfit/issues/517)) ([fab2bec](https://github.com/AlexsJones/llmfit/commit/fab2bec907db77ca63866a6c980bf8418d804b88)), closes [#515](https://github.com/AlexsJones/llmfit/issues/515)

## [0.9.16](https://github.com/AlexsJones/llmfit/compare/v0.9.15...v0.9.16) (2026-04-27)


### Features

* add community benchmarks view from localmaxxing.com ([4d0ffa7](https://github.com/AlexsJones/llmfit/commit/4d0ffa79d8f332b51d5fbc990815612bf03b5069))
* add hardware picker to benchmarks view (H key) ([21661b2](https://github.com/AlexsJones/llmfit/commit/21661b21fb28261502f915b62eb9691a5f837dcf))
* add vLLM inference provider ([bfb0bb2](https://github.com/AlexsJones/llmfit/commit/bfb0bb202b97bc0fc97973b3688e54e83397a195)), closes [#501](https://github.com/AlexsJones/llmfit/issues/501)
* **download:** add --output-dir flag to override cache path ([#499](https://github.com/AlexsJones/llmfit/issues/499)) ([f70409f](https://github.com/AlexsJones/llmfit/commit/f70409fc1c904100bc941c89aeea62b4fa8594b2))


### Bug Fixes

* add debug warning when util &gt; 1.0 in GPU-mode VRAM pressure calculation ([#513](https://github.com/AlexsJones/llmfit/issues/513)) ([92d0f63](https://github.com/AlexsJones/llmfit/commit/92d0f63c54669f4381be32c75c2b86532b1b0618)), closes [#496](https://github.com/AlexsJones/llmfit/issues/496)
* deduplicate AMD GPUs detected by both ROCm and Vulkan (fixes [#139](https://github.com/AlexsJones/llmfit/issues/139)) ([#314](https://github.com/AlexsJones/llmfit/issues/314)) ([43dd9fc](https://github.com/AlexsJones/llmfit/commit/43dd9fc075ec20fca1fcd1598c9bf98df32ad9de))
* match leaderboard API nested response schema ([4282eb4](https://github.com/AlexsJones/llmfit/commit/4282eb46331c8d14454baafe1af1760f28e80f48))

## [0.9.15](https://github.com/AlexsJones/llmfit/compare/v0.9.14...v0.9.15) (2026-04-25)


### Features

* add DeepSeek V4 model family ([#502](https://github.com/AlexsJones/llmfit/issues/502)) ([0fd800f](https://github.com/AlexsJones/llmfit/commit/0fd800fa1b6bbd7c3280c84b0c384e12ddf55539)), closes [#500](https://github.com/AlexsJones/llmfit/issues/500)
* make key 'g' cycle top/bottom ([#454](https://github.com/AlexsJones/llmfit/issues/454)) ([2ccb9bc](https://github.com/AlexsJones/llmfit/commit/2ccb9bc29c71ea333cde4ca99c6dbbf8cd29c6d4))


### Bug Fixes

* MoE min_vram regression — all experts must be in memory ([#497](https://github.com/AlexsJones/llmfit/issues/497)) ([b8f37fd](https://github.com/AlexsJones/llmfit/commit/b8f37fd8131d1cccc23f114634afd1b49cc89cfc))
* resolve a build warning for an unused variable ([#331](https://github.com/AlexsJones/llmfit/issues/331)) ([1e68d80](https://github.com/AlexsJones/llmfit/commit/1e68d8081140fa4303fb608180c19b0825f9a760))

## [0.9.14](https://github.com/AlexsJones/llmfit/compare/v0.9.13...v0.9.14) (2026-04-23)


### Features

* add Qwen3.6 models and refresh model database with discovery ([5397a8a](https://github.com/AlexsJones/llmfit/commit/5397a8ab43819f9c371588be06c3e2165118d78c))


### Bug Fixes

* MoE GPU-mode uses active params + UD-quant support ([#475](https://github.com/AlexsJones/llmfit/issues/475)) ([2768c0f](https://github.com/AlexsJones/llmfit/commit/2768c0f6dfc20bb90e9289188b7aa468f6c6e369))
* populate fit.installed in llmfit fit CLI command ([#491](https://github.com/AlexsJones/llmfit/issues/491)) ([2b67d2e](https://github.com/AlexsJones/llmfit/commit/2b67d2ee5e24a5dc2ad2ff84b2c954a8832d5070))

## [0.9.13](https://github.com/AlexsJones/llmfit/compare/v0.9.12...v0.9.13) (2026-04-22)


### Bug Fixes

* installed flag always false for models without explicit mapping ([#481](https://github.com/AlexsJones/llmfit/issues/481)) ([#492](https://github.com/AlexsJones/llmfit/issues/492)) ([7d4937f](https://github.com/AlexsJones/llmfit/commit/7d4937fc2564b7c9466fce060e09bd1b66d9a959))

## [0.9.12](https://github.com/AlexsJones/llmfit/compare/v0.9.11...v0.9.12) (2026-04-22)


### Features

* add --tool-use filter flag to fit command ([#465](https://github.com/AlexsJones/llmfit/issues/465)) ([0ffd8cd](https://github.com/AlexsJones/llmfit/commit/0ffd8cd010e013000c4147d5b99d22418ad41633))
* add Filter Popup (F key) with range filters, sort direction, and fit ([#458](https://github.com/AlexsJones/llmfit/issues/458)) ([d182cc3](https://github.com/AlexsJones/llmfit/commit/d182cc34b097dea79887cf30de690d4e39206814))


### Bug Fixes

* **ci:** pass crates.io token via env, not argv ([#487](https://github.com/AlexsJones/llmfit/issues/487)) ([7be5edb](https://github.com/AlexsJones/llmfit/commit/7be5edb3448e4a809ebe7a342189d762e09b1af7))
* fall back to 127.0.0.1 when localhost Ollama detection fails (fixes [#57](https://github.com/AlexsJones/llmfit/issues/57)) ([#319](https://github.com/AlexsJones/llmfit/issues/319)) ([126f81a](https://github.com/AlexsJones/llmfit/commit/126f81a141237808bb2db7e60bce3db02025f9f5))
* parse GPU name from rocm-smi output correctly ([#479](https://github.com/AlexsJones/llmfit/issues/479)) ([fca1984](https://github.com/AlexsJones/llmfit/commit/fca1984bc3bb8c8f4cac9e1965283a5071212aed))
* **providers:** prevent argument injection in hf and docker subprocess calls ([#329](https://github.com/AlexsJones/llmfit/issues/329)) ([319c0fe](https://github.com/AlexsJones/llmfit/commit/319c0fe5b2d0845f1279e4440772be17ed8e1de3))
* synchronize Cargo.lock and update release workflow to keep it in sync ([#477](https://github.com/AlexsJones/llmfit/issues/477)) ([9a99e69](https://github.com/AlexsJones/llmfit/commit/9a99e699af529c14577b12addd64b4492312c1f8)), closes [#476](https://github.com/AlexsJones/llmfit/issues/476)
* **tui:** bind auto-spawned dashboard to loopback by default ([#330](https://github.com/AlexsJones/llmfit/issues/330)) ([6513a69](https://github.com/AlexsJones/llmfit/commit/6513a697d1ba4fb52a0103ff56a5790c631978c5))
* **tui:** fix filter popup Esc-revert, cursor alignment, and code cleanup ([5591b90](https://github.com/AlexsJones/llmfit/commit/5591b90946526713e852dec6f5123034051fec7f))
* **tui:** use theme background for popups and wrap GGUF source URLs ([#407](https://github.com/AlexsJones/llmfit/issues/407)) ([eb5c9bf](https://github.com/AlexsJones/llmfit/commit/eb5c9bfc5c24f7c38428c830adbd11168d96e1b3)), closes [#344](https://github.com/AlexsJones/llmfit/issues/344)

## [0.9.11](https://github.com/AlexsJones/llmfit/compare/v0.9.10...v0.9.11) (2026-04-19)


### Bug Fixes

* MoE GPU-mode TPS uses full model size + expert-count-scaled overhead ([#464](https://github.com/AlexsJones/llmfit/issues/464)) ([e1f3245](https://github.com/AlexsJones/llmfit/commit/e1f32459ee6d2f486064cbda045b9a4faaabbc69))
* read LM Studio download progress from streaming POST response ([#467](https://github.com/AlexsJones/llmfit/issues/467)) ([3b90e2c](https://github.com/AlexsJones/llmfit/commit/3b90e2cf0f4044d556eb3235b8bd0fbbdc905008)), closes [#463](https://github.com/AlexsJones/llmfit/issues/463)
* replace unreachable!() with graceful fallback in estimate_tps_with_gpu ([#459](https://github.com/AlexsJones/llmfit/issues/459)) ([fd1b233](https://github.com/AlexsJones/llmfit/commit/fd1b233de7f52c4d33592a6a2b511da232e5e7d4))

## [0.9.10](https://github.com/AlexsJones/llmfit/compare/v0.9.9...v0.9.10) (2026-04-18)


### Features

* add --csv flag for spreadsheet-compatible model fit export ([#456](https://github.com/AlexsJones/llmfit/issues/456)) ([48227e6](https://github.com/AlexsJones/llmfit/commit/48227e6032aaa9448d14283f4cab3ba1fb5eb099))
* **tui:** add Download Manager view (D) ([c86dbb4](https://github.com/AlexsJones/llmfit/commit/c86dbb491496ff62c3fe2e338fe90a86cba7622e))
* **tui:** show VRAM estimate at common context sizes in detail view ([#320](https://github.com/AlexsJones/llmfit/issues/320)) ([3e6f1ab](https://github.com/AlexsJones/llmfit/commit/3e6f1abc0cf5d64520973709d5652075d1e55ac5))


### Bug Fixes

* use Win32_PhysicalMemory to detect AMD APU VRAM on Windows (fixes [#194](https://github.com/AlexsJones/llmfit/issues/194)) ([#452](https://github.com/AlexsJones/llmfit/issues/452)) ([1da9aac](https://github.com/AlexsJones/llmfit/commit/1da9aac210c554a287998bfe8500f790eba4b782))

## [0.9.9](https://github.com/AlexsJones/llmfit/compare/v0.9.8...v0.9.9) (2026-04-17)


### Bug Fixes

* correct Llama 4 Maverick metadata (expert count, context length) ([#446](https://github.com/AlexsJones/llmfit/issues/446)) ([ac5b115](https://github.com/AlexsJones/llmfit/commit/ac5b115988369de21cdef91b00ea28076dce7c4a))
* improve CPU-only TPS accuracy & add cross-platform Advanced Config modal ([#455](https://github.com/AlexsJones/llmfit/issues/455)) ([7e0918b](https://github.com/AlexsJones/llmfit/commit/7e0918b9baeb677f679f13ea754e43d244fe81c3))

## [0.9.8](https://github.com/AlexsJones/llmfit/compare/v0.9.7...v0.9.8) (2026-04-13)


### Bug Fixes

* **ci:** use RELEASE_TOKEN in release-please to trigger downstream workflows ([f216779](https://github.com/AlexsJones/llmfit/commit/f21677927b9d05a348a6ed33da6faaf832f9d82c))
* prefer discrete GPU over integrated on hybrid systems ([#303](https://github.com/AlexsJones/llmfit/issues/303)) ([2707b54](https://github.com/AlexsJones/llmfit/commit/2707b54b2ee60ee2132512a4534c1db082701b83))

## [0.9.7](https://github.com/AlexsJones/llmfit/compare/v0.9.6...v0.9.7) (2026-04-13)


### Features

* add --force-runtime flag to override automatic runtime selection ([c31099f](https://github.com/AlexsJones/llmfit/commit/c31099f7de99091f3a5c5593a453df89d3feafbc))
* add --memory flag to override GPU VRAM autodetection ([9a35a76](https://github.com/AlexsJones/llmfit/commit/9a35a76470b02f51f572fb88e674974f86329b08))
* add --memory flag to override GPU VRAM autodetection ([b6e323d](https://github.com/AlexsJones/llmfit/commit/b6e323d72816f8f5920943c573c3897b12ce5b49))
* add "Sort by Release Date" to TUI ([f371cd8](https://github.com/AlexsJones/llmfit/commit/f371cd88f6bf2eae61b0fbbc6c3d4b3454ea0fb2))
* add "Sort by Release Date" to TUI ([d55670a](https://github.com/AlexsJones/llmfit/commit/d55670aba8ae0f359f8ee0c85444a5bab16d383d))
* add 15 popular models from HuggingFace ([eb886d6](https://github.com/AlexsJones/llmfit/commit/eb886d643dcef083650d980ef8c36114310f5110))
* Add 15 popular models from HuggingFace (33→48 models) ([e0c0b52](https://github.com/AlexsJones/llmfit/commit/e0c0b528b5e6bed070c2f2f4123f6cddc70b5125))
* add 18 new models to database (Feb 2026) ([8612c0e](https://github.com/AlexsJones/llmfit/commit/8612c0e4feaa072eadaa44ca492d4c1e226fe091))
* add 18 new models to database (Feb 2026) ([1c040f2](https://github.com/AlexsJones/llmfit/commit/1c040f280fe2a790211c1b6a4d302a3f7edfd00b))
* add 18 new models to database (Feb 2026) — conflict-resolved ([0970845](https://github.com/AlexsJones/llmfit/commit/097084568841b5c2a8c7f17f65042612e7e5b68c))
* add AWQ/GPTQ support with vLLM inference runtime ([e398490](https://github.com/AlexsJones/llmfit/commit/e3984901f9d5eadff02c3683f8ebd646aededf4c))
* add AWQ/GPTQ support with vLLM inference runtime ([d7f2f11](https://github.com/AlexsJones/llmfit/commit/d7f2f11bb29efb56f500669c2e4e820ce6e93d4e))
* add Catppuccin color themes to TUI ([f05d299](https://github.com/AlexsJones/llmfit/commit/f05d299f541866721f061ed8599fd9e3072b4f60))
* add Catppuccin color themes to TUI ([104b281](https://github.com/AlexsJones/llmfit/commit/104b28130ba246167fb6019d531548090128b189))
* add CLI/TUI model diff compare workflow ([3cab068](https://github.com/AlexsJones/llmfit/commit/3cab068f83fda006988631fe4134501db42edde1))
* add context-length cap for memory estimation (--max-context) ([0d9a0ef](https://github.com/AlexsJones/llmfit/commit/0d9a0ef8ec7f2d200671ebbb9207b9e1d7bf25a2))
* Add Docker containerization support ([42ff7d3](https://github.com/AlexsJones/llmfit/commit/42ff7d35177397c0b95556873b600505e7555032))
* Add Docker containerization support ([fbb483c](https://github.com/AlexsJones/llmfit/commit/fbb483cf4c3db735eac8248cd16413dd67c31810))
* add Docker Model Runner as a runtime provider ([2a84138](https://github.com/AlexsJones/llmfit/commit/2a84138b001ec02768ebb9afc69274b023c29344))
* add fuzzy search with space-separated terms ([89bbe3d](https://github.com/AlexsJones/llmfit/commit/89bbe3df09b40dfee195c5dde4974f36f19d6598))
* add fuzzy search with space-separated terms ([3d967f4](https://github.com/AlexsJones/llmfit/commit/3d967f4ec72f3126863dc27b43d49000fd0686fa)), closes [#65](https://github.com/AlexsJones/llmfit/issues/65)
* add GGUF download source enrichment for models ([967253b](https://github.com/AlexsJones/llmfit/commit/967253b6098dbb45596999eca5e4a1483d70a8c9))
* add Google Gemma 4 models and fix Gemma 3 capabilities ([2cb21c3](https://github.com/AlexsJones/llmfit/commit/2cb21c36b97a64546eb8cc6b49318d42f672a1fb))
* add homebrew tap support and update release workflow ([67b6fcf](https://github.com/AlexsJones/llmfit/commit/67b6fcf371255a72538b9aca2912f666b8627978))
* add InferenceRuntime enum and MLX quantization support ([186b95e](https://github.com/AlexsJones/llmfit/commit/186b95ef1a76ea3ddb6b6b27d7ae0046ba38fe96))
* add license filter for models ([5bb4b1e](https://github.com/AlexsJones/llmfit/commit/5bb4b1e6ea6d9bcd426760090459e301bb38206c))
* add license filter for models ([#186](https://github.com/AlexsJones/llmfit/issues/186)) ([33bd708](https://github.com/AlexsJones/llmfit/commit/33bd70870f682d760cdc8d1d3dc7aed9ea2cd978))
* add Liquid AI LFM2/LFM2.5 models ([83d038a](https://github.com/AlexsJones/llmfit/commit/83d038a54e7c62d740a4423261959b9c5c49dd33))
* add max-context cap for memory estimation ([afba24a](https://github.com/AlexsJones/llmfit/commit/afba24a2a0abb71f8ce7005ed3dcfcccdeb64f0e))
* add MiniMax-M2.7 to curated model database ([a5cda9b](https://github.com/AlexsJones/llmfit/commit/a5cda9bbc23e3b494bf67677f7a123601ecff1e3))
* add MiniMax-M2.7 to curated model database ([7c16b9d](https://github.com/AlexsJones/llmfit/commit/7c16b9d5ee70b321257d74cb8ca4d5b76cda0c55))
* add MlxProvider for MLX model detection and downloads ([7f0ee24](https://github.com/AlexsJones/llmfit/commit/7f0ee2447d1062e5b737ed80777e14d4d207aa5f))
* add model comparison workflow (llmfit diff) and TUI compare mode ([76a8a9c](https://github.com/AlexsJones/llmfit/commit/76a8a9c80debad9c2b6df140dc7191778615ff39))
* add Ollama mappings for new models ([5c97e75](https://github.com/AlexsJones/llmfit/commit/5c97e75df38a5ba429c3a6ab60dd37b560514d9d))
* add Qwen 3.5 series (397B/122B/35B/27B) to model database ([0d550f3](https://github.com/AlexsJones/llmfit/commit/0d550f3ae5415e8e84eb40941fef32cfaa9cd0ab))
* add Qwen3-Coder-Next (80B MoE) and Qwen 3.5 Ollama mappings ([5f8a640](https://github.com/AlexsJones/llmfit/commit/5f8a640924e505cc889eb7a4afc70f45c2614474))
* add Qwen3.5 small model series (0.8B, 2B, 4B, 9B) ([21cc8ac](https://github.com/AlexsJones/llmfit/commit/21cc8ac2585ac0a5bf4b6ceab99f27c07e51acde))
* add Qwen3.5 small model series (0.8B, 2B, 4B, 9B) ([226c5c2](https://github.com/AlexsJones/llmfit/commit/226c5c2d805caa2418992a1a6076f1a2125b6082))
* add runtime model database updates from HuggingFace ([d465fb9](https://github.com/AlexsJones/llmfit/commit/d465fb9ddce782ff094b3f462aee71852b02c807))
* add runtime model database updates from HuggingFace ([1e20ec3](https://github.com/AlexsJones/llmfit/commit/1e20ec3f13711d7fd7538619fe5347f209c6ec46))
* add tensor parallelism awareness for multi-GPU model fitting ([e8a6001](https://github.com/AlexsJones/llmfit/commit/e8a60011d751bffebe65a22ef164a11f8c22a691))
* add tensor parallelism awareness for multi-GPU model fitting ([5d99ec2](https://github.com/AlexsJones/llmfit/commit/5d99ec2274ef8e18f2cbf0960bd525f0cf480a05))
* add theme switcher with 6 color schemes for TUI ([317c44a](https://github.com/AlexsJones/llmfit/commit/317c44a8b06067938117b4ad7b60e7e3298ab08c))
* add theme switcher with 6 color schemes for TUI ([89bd63f](https://github.com/AlexsJones/llmfit/commit/89bd63f0166f7af4c00efd643fe956c58f824f4f))
* add tok/s column sorting in TUI ([41b7358](https://github.com/AlexsJones/llmfit/commit/41b73580e7d16c6a318256ae1bd1bc704b457608))
* add tok/s sort option to TUI sort cycle and table header highlighting ([82bb526](https://github.com/AlexsJones/llmfit/commit/82bb526b00f19affb7a53edac56f4d4fd73311da))
* add use-case popup filter and use-case search matching to TUI ([4b1aeab](https://github.com/AlexsJones/llmfit/commit/4b1aeabb1f3961e4e749714ef6d00b2b41d76a27))
* add use-case popup filter to TUI ([4b0f452](https://github.com/AlexsJones/llmfit/commit/4b0f452fa89974aa16a1c311d23890660e1eb74f))
* added arc support ([0d5e991](https://github.com/AlexsJones/llmfit/commit/0d5e991a53c3b9ec3639e32318ca74ef70fba5d0))
* added in vim like bindings ([a3df62f](https://github.com/AlexsJones/llmfit/commit/a3df62f5654f3e681597266b4f29c7f6c7ac005e))
* added LM studio ([70e60cc](https://github.com/AlexsJones/llmfit/commit/70e60ccd03a02d11cdc5aa818a168934530ee44d))
* added logo ([86dd737](https://github.com/AlexsJones/llmfit/commit/86dd7379ae62e74df477f70287b24587d144a09c))
* added moe ([4723782](https://github.com/AlexsJones/llmfit/commit/47237826f1cd4428a8ebe0fac1a8b5369796882a))
* added Rest API ([2adb593](https://github.com/AlexsJones/llmfit/commit/2adb593f44137b479b6fc986b9083d373afad530))
* adding ollama as supported provider ([e198c17](https://github.com/AlexsJones/llmfit/commit/e198c17687ee5560384655df52e6bb9dcd5ba014))
* adding release please ([c68914a](https://github.com/AlexsJones/llmfit/commit/c68914a5cd51c270585d74c2ffe8fdddc157c34a))
* adding serve capabilities ([503229e](https://github.com/AlexsJones/llmfit/commit/503229e5664d7bdd065ff84a30a36dae27a55e90))
* **api:** add capabilities, license, supports_tp to model response and new endpoints ([c02e189](https://github.com/AlexsJones/llmfit/commit/c02e189f676a44714bddaff1348495d7863a860d))
* append (WSL) to RAM label in tui when running under WSL ([aa30f57](https://github.com/AlexsJones/llmfit/commit/aa30f57276535c84bf835c2b87576644cc69af81))
* availability filter [a] in TUI (All / GGUF Avail / Installed) ([9356150](https://github.com/AlexsJones/llmfit/commit/935615088a05b779d0e762603d994bed6f20c246))
* bandwidth-based tok/s estimation for known GPUs ([0c552b4](https://github.com/AlexsJones/llmfit/commit/0c552b470316b05976f42c85f7f1472194395a66))
* bandwidth-based tok/s estimation for known GPUs ([1c34600](https://github.com/AlexsJones/llmfit/commit/1c34600e9d5388badbdcd5f72e4848ef094e08a8))
* cargo fmt ([cde77e3](https://github.com/AlexsJones/llmfit/commit/cde77e35019e1e0e998dcaf2a6fc4b228918a909))
* caught some unavailable models on ollama ([1d68bba](https://github.com/AlexsJones/llmfit/commit/1d68bbabf6a0416cf8e619ad38cc56be95f58a32))
* caught some unavailable models on ollama ([6aaba9f](https://github.com/AlexsJones/llmfit/commit/6aaba9fd77abda41e6867d4d4018843f875bada7))
* **cli:** add --sort for fit output ([8114446](https://github.com/AlexsJones/llmfit/commit/8114446eafbfcad27e3ae097a4a42f4d20aa9e7a))
* **cli:** add `--sort` option for fit output ([3a06877](https://github.com/AlexsJones/llmfit/commit/3a06877ce2908bec4c39b5164a1d5a0c908b9d7a))
* crate version yank skip rebuild ([b1b0628](https://github.com/AlexsJones/llmfit/commit/b1b06286723f4e21bea81ce7f2602f12ca0fe5bb))
* **dashboard:** ship embedded web UI and auto-start from CLI ([96bd53c](https://github.com/AlexsJones/llmfit/commit/96bd53c9c9cee7cbe1d1f1ee08559b6fc4b9aa2e))
* **dashboard:** ship embedded web UI and auto-start it from CLI ([37dd0db](https://github.com/AlexsJones/llmfit/commit/37dd0dbca4b9dab1fc96597a64751a4267128fa6))
* detect installed Ollama models and support pulling from TUI ([dc05d5a](https://github.com/AlexsJones/llmfit/commit/dc05d5a41222f5961ac544c22e43b7398431c901))
* expand mlx-community model mappings ([7a9e917](https://github.com/AlexsJones/llmfit/commit/7a9e9171b10776186f9a259450d0ebfc8d0b2f40))
* expand mlx-community model mappings and improve fallback heuristic ([5e8a676](https://github.com/AlexsJones/llmfit/commit/5e8a6769ccf3ae0754b80b91fb39e35144553b74)), closes [#260](https://github.com/AlexsJones/llmfit/issues/260)
* fix for qwen3_5moe ([455709f](https://github.com/AlexsJones/llmfit/commit/455709f2fb7524b2c62b455c56454ce3897669c6))
* fixed regression in LM studio model list ([cbeccfa](https://github.com/AlexsJones/llmfit/commit/cbeccfa60546b1c150d103621470bf5b2b32bb23))
* fixed regression in LM studio model list ([734f556](https://github.com/AlexsJones/llmfit/commit/734f55620114d3b14d17bf24e399b7ace4f7e4c3))
* fixed up skill ([38c8350](https://github.com/AlexsJones/llmfit/commit/38c8350e2f78c3207a2c40e1079267db6b707809))
* fixed up skill ([368fa47](https://github.com/AlexsJones/llmfit/commit/368fa4728064238c377fb0632c7fc83a5688c504))
* fixing vram on apple bug ([bd59767](https://github.com/AlexsJones/llmfit/commit/bd59767f2b6fdc7dfd1352cbb12f7dff17e18d8b))
* fixing vram on apple bug ([db666b6](https://github.com/AlexsJones/llmfit/commit/db666b66d4087ecf1b62c1c9f582e765007eb30b))
* fixing vram on apple bug ([c26d13b](https://github.com/AlexsJones/llmfit/commit/c26d13bde0990549c6bc1c3f090d1935c23ec036))
* fixing vram on apple bug ([085cd25](https://github.com/AlexsJones/llmfit/commit/085cd25a3730c762ed21e9e0c4570ecb6a17323c))
* GGUF download source enrichment (unsloth & bartowski) ([6e3c828](https://github.com/AlexsJones/llmfit/commit/6e3c828123009f76088265a11143e8d5e7474ed0))
* hardware simulation mode and CLI hardware overrides ([002eeff](https://github.com/AlexsJones/llmfit/commit/002eeff6acf61970c2d3bf471a3b2a1afc3f82ba)), closes [#322](https://github.com/AlexsJones/llmfit/issues/322)
* **hardware:** detect NVIDIA Tegra/Grace Blackwell unified memory via ATS (rebased [#93](https://github.com/AlexsJones/llmfit/issues/93)) ([b454697](https://github.com/AlexsJones/llmfit/commit/b45469797c502d76e2b904ccf50cdb7d5781a648))
* **hardware:** detect NVIDIA Tegra/Grace Blackwell unified memory via ATS addressing mode ([dd1caa6](https://github.com/AlexsJones/llmfit/commit/dd1caa60d4ffec36605f7fed9c2cb77ed3b1a23e))
* improvements based on [#12](https://github.com/AlexsJones/llmfit/issues/12) ([7997525](https://github.com/AlexsJones/llmfit/commit/7997525fc248e3cff845b47a14afac9694401cd0))
* increased model count ([5be634f](https://github.com/AlexsJones/llmfit/commit/5be634f3455f68db82aedc21357df243b60c8688))
* increment version ([d88bd3f](https://github.com/AlexsJones/llmfit/commit/d88bd3fecd7e11cca97c6426fa4aee9ec18df760))
* installed status for lm studio ([a1cad30](https://github.com/AlexsJones/llmfit/commit/a1cad30d1cf6cf08f70833347487c480d5650350))
* mlx disablement on non-apple hw ([c26bb49](https://github.com/AlexsJones/llmfit/commit/c26bb4927ab54dde78fab827125c5901ea6f238f))
* MLX-native inference support for Apple Silicon ([993599e](https://github.com/AlexsJones/llmfit/commit/993599e873141a6676ce5ce428cec15fb5201c46))
* model detail modal + Ollama download in desktop app ([c92053f](https://github.com/AlexsJones/llmfit/commit/c92053f38e6980a6c0865d0637a49deeb9c0ce52))
* model detail modal + Ollama download in desktop app ([dacae77](https://github.com/AlexsJones/llmfit/commit/dacae77a01739ef50084e3ab6847b7bf5850f290))
* overall to the scoring system ([e689f95](https://github.com/AlexsJones/llmfit/commit/e689f95e90bda56a417067cfe7033c7b0b865ad1))
* overall to the scoring system ([d411a8d](https://github.com/AlexsJones/llmfit/commit/d411a8dd773fa2fdd813f3c5026074e76b85b51e))
* overall to the scoring system ([5ca2467](https://github.com/AlexsJones/llmfit/commit/5ca24672985c7a1dfc02334a537dce6f4c8502c4))
* persist filter state across sessions ([#430](https://github.com/AlexsJones/llmfit/issues/430)) ([c3962e4](https://github.com/AlexsJones/llmfit/commit/c3962e4645ccd11bb6a5e5cf23a222e30c98af93))
* persist theme selection to ~/.config/llmfit/theme ([765c87b](https://github.com/AlexsJones/llmfit/commit/765c87b4ea8352855fc3774f8af06dee70ae63ed))
* **plan:** KV cache fidelity + KV quant flag + TurboQuant gating ([1f1b6fc](https://github.com/AlexsJones/llmfit/commit/1f1b6fc25a04ee1cd7881f4e33becf7d24b95588))
* plumbing 2 ([209f9d4](https://github.com/AlexsJones/llmfit/commit/209f9d4b1814923b88bf928d78fdc193fb37af9b))
* plumbing 2 ([ef18920](https://github.com/AlexsJones/llmfit/commit/ef1892077c9bbd66bf47b20153ff6f5afff688fa))
* pull functionality ([e193d56](https://github.com/AlexsJones/llmfit/commit/e193d5678f954cc56c6a7d1aefad1616b29d5607))
* rebased ([ac6639d](https://github.com/AlexsJones/llmfit/commit/ac6639d1b44e8f100fadd2044d571c66c9988d06))
* release plumbing ([648a8c1](https://github.com/AlexsJones/llmfit/commit/648a8c179cf174f69a207bc7683e5f6be1a037da))
* release plumbing ([8215ba8](https://github.com/AlexsJones/llmfit/commit/8215ba80304b77765faac85dba3525434f2f3b9b))
* removed exact model count as it increases so often ([1a19c42](https://github.com/AlexsJones/llmfit/commit/1a19c4258f9435b2d3eddaaade7fd008b49919eb))
* reworked available models for download ([6d56b03](https://github.com/AlexsJones/llmfit/commit/6d56b03c128ba720787e9562270dae6c722fae2c))
* show approximate disk space usage ([#134](https://github.com/AlexsJones/llmfit/issues/134)) ([#304](https://github.com/AlexsJones/llmfit/issues/304)) ([32c835b](https://github.com/AlexsJones/llmfit/commit/32c835bf421a6b0727b32e5c2cddba8bf5c581e7))
* split-pane TUI detail view for GGUF downloads ([de5c53b](https://github.com/AlexsJones/llmfit/commit/de5c53bd6426d51292421460923ea4a478f2da94))
* support for windows vulkan ([5a76a8f](https://github.com/AlexsJones/llmfit/commit/5a76a8f956e4732fe4fe9a12020d34f476aa9f0b))
* supporting 94 models ([c68d7aa](https://github.com/AlexsJones/llmfit/commit/c68d7aad0c0a00cbf12ae49c7325d74bf0142fce))
* surface runtime info in TUI, CLI, and JSON output ([4bdb4c4](https://github.com/AlexsJones/llmfit/commit/4bdb4c487eeb4e2fdd1943bfe21c38081a1a1091))
* This PR is to add the ability for capabilities to be described ([d09b8fe](https://github.com/AlexsJones/llmfit/commit/d09b8febedd0a00e1ddf10fe72db2ab8ffa07210))
* This PR is to add the ability for capabilities to be described ([05768af](https://github.com/AlexsJones/llmfit/commit/05768af09db6323c38f2ac5b27a6214a8133caf1))
* TUI GGUF downloads section, default enrichment, caching ([e7dc0b2](https://github.com/AlexsJones/llmfit/commit/e7dc0b216cc2c22af1478fd91cfd0a711009ad9c))
* **tui:** add runtime/backend filter and help popup (v0.9.2) ([ad28138](https://github.com/AlexsJones/llmfit/commit/ad28138aa1362f74caae9c6e3abaf49ba68b7480))
* update key bindings and add Catppuccin themes to README files ([c41567b](https://github.com/AlexsJones/llmfit/commit/c41567b5caa7bf6ad49a27bd5e4e895861038e74))
* updated build actions ([59a8af4](https://github.com/AlexsJones/llmfit/commit/59a8af4ced8492da2fbcd954ec4b300ba40cf954))
* updated demo ([6c32985](https://github.com/AlexsJones/llmfit/commit/6c32985afc222c000a5644ffaaf4e92d16612900))
* updated demo ([42fbeeb](https://github.com/AlexsJones/llmfit/commit/42fbeeb114753045ab10af8e1e6c9d0d201dd1ef))
* updated images ([20205bd](https://github.com/AlexsJones/llmfit/commit/20205bd4a715f3c65337836fbb507774fae47123))
* updated models ([dfb9afb](https://github.com/AlexsJones/llmfit/commit/dfb9afb743d7af348e86f50892584822d85b3ce5))
* updated sorting and new models ([744465d](https://github.com/AlexsJones/llmfit/commit/744465d2a88b04ea737feec75dd000d9347cf9ab))
* updated tui to support multiple providers better and also multiple GPU suppor ([df338e8](https://github.com/AlexsJones/llmfit/commit/df338e821b7b411b30b6be87b6a4202ced68dabb))
* updated urls ([ce14772](https://github.com/AlexsJones/llmfit/commit/ce14772ea18da3d7b0d6554ed678295fd97d5480))
* updated version ([4d1542b](https://github.com/AlexsJones/llmfit/commit/4d1542b582d0af0a7bdb22f9d89573cc703d5f36))
* updating models ([d4f0a0e](https://github.com/AlexsJones/llmfit/commit/d4f0a0e19a880d95e500bf524bf52c39b56c0bf1))
* version bump llama.cpp added ([1643214](https://github.com/AlexsJones/llmfit/commit/1643214e6547e8ea98a84f56101079ed6c737b18))
* version bump llama.cpp added ([319fb90](https://github.com/AlexsJones/llmfit/commit/319fb9065f88480d12ba9a3d15c3d323c1f448d4))
* **web:** add 10 color themes from TUI theme.rs ([ab14a06](https://github.com/AlexsJones/llmfit/commit/ab14a0699e6765b8cb5c03a4a16a105064c296c7))
* **web:** add side-by-side model comparison view ([ccb56e9](https://github.com/AlexsJones/llmfit/commit/ccb56e964b875c8db31946d5021bd646dc194ac6))
* **web:** replace freeform limit input with preset options ([96981ec](https://github.com/AlexsJones/llmfit/commit/96981ec65e2642dd8acc91ff7d512d2b4cb3b90f))
* **web:** split App.jsx into Header, SystemPanel, FilterBar, ModelTable, DetailPanel ([950e620](https://github.com/AlexsJones/llmfit/commit/950e6208bfd14673820f5acbeb7ed193f9af7a1d))
* working on v0.8.1 ([696107d](https://github.com/AlexsJones/llmfit/commit/696107d7650fd6351dc53c7f00bef7c6065a1172))
* workspace restructure + Tauri desktop app ([83ded27](https://github.com/AlexsJones/llmfit/commit/83ded27beac5b1772dc9aeec7638d44a4b6e5fb1))
* workspace restructure + Tauri desktop app ([3fb10d0](https://github.com/AlexsJones/llmfit/commit/3fb10d019f828662a21619d282570cf32f0ff833))


### Bug Fixes

* accept OLLAMA_HOST values without URL scheme ([#166](https://github.com/AlexsJones/llmfit/issues/166)) ([c647b13](https://github.com/AlexsJones/llmfit/commit/c647b13787edd4031713196d42d42543139bae51))
* add --local flag to install script and improve fallback logic ([1e5cda1](https://github.com/AlexsJones/llmfit/commit/1e5cda1a0d4bd7ca2bf50abd553599f57efdbb78))
* add --local flag to install script and improve fallback logic ([23b500e](https://github.com/AlexsJones/llmfit/commit/23b500e8bccd25b0bf7ee61a5181147f67e4ca39)), closes [#56](https://github.com/AlexsJones/llmfit/issues/56)
* add AMD RX 9060 series to VRAM estimation database ([1c02da8](https://github.com/AlexsJones/llmfit/commit/1c02da8b74af828d7f1386a261073063647efd88))
* add AMD RX 9060 series to VRAM estimation database ([9305810](https://github.com/AlexsJones/llmfit/commit/9305810ad7f226abb8f2513e04c45250f827b8e6)), closes [#55](https://github.com/AlexsJones/llmfit/issues/55)
* add download confirmation to prevent accidental pulls ([9e4e881](https://github.com/AlexsJones/llmfit/commit/9e4e881c78fae4e9bd5f5adbdb9dfbf8302ed56c))
* add download confirmation to prevent accidental pulls ([#95](https://github.com/AlexsJones/llmfit/issues/95)) ([5b56d74](https://github.com/AlexsJones/llmfit/commit/5b56d74e422130718e635e4d82533d8e34f7670d))
* address AlexsJones review comments ([587252b](https://github.com/AlexsJones/llmfit/commit/587252b0124e3e7f8dbcdce4ac1ccaed7c7345de))
* address Copilot review comments ([74384c9](https://github.com/AlexsJones/llmfit/commit/74384c9ea7d1215aa38c455b20d5005bbc8fa361))
* AMD Ryzen AI unified memory detection and --memory override ([174fc10](https://github.com/AlexsJones/llmfit/commit/174fc10bb93cccdfdfd1441cc9bc60d1682d1088))
* AMD unified memory detection and --memory override ([c763717](https://github.com/AlexsJones/llmfit/commit/c76371785ef93761eb411b67acc3202832ab9ce5)), closes [#89](https://github.com/AlexsJones/llmfit/issues/89) [#91](https://github.com/AlexsJones/llmfit/issues/91)
* bump Dockerfile Rust version to 1.88 for dependency compatibility ([c25c02e](https://github.com/AlexsJones/llmfit/commit/c25c02e00b2709df869fff8fd72c9419303d504a))
* bump Dockerfile Rust version to 1.88 for dependency compatibility ([6304ba0](https://github.com/AlexsJones/llmfit/commit/6304ba0c19f804353afc4c6e79c29cee8a51ae70))
* cap default estimation context at 8192 tokens ([#311](https://github.com/AlexsJones/llmfit/issues/311)) ([4699ecb](https://github.com/AlexsJones/llmfit/commit/4699ecbf787a89ae4fbb246896b0c9711f893a12))
* **ci:** switch release-please to simple type for workspace version inheritance ([48be41a](https://github.com/AlexsJones/llmfit/commit/48be41aad22582a5921f0e1b3db2918aaad92950))
* **cli:** default auto dashboard host to 0.0.0.0 ([4553fbf](https://github.com/AlexsJones/llmfit/commit/4553fbfa2a00b36626b3feef49b2fbfe9d888ae0))
* **cli:** make dashboard auto-start controllable and self-cleaning ([0c51ad6](https://github.com/AlexsJones/llmfit/commit/0c51ad6ea192da24641aade2c7d22aec0db0d5db))
* correct crates.io metadata and prepare for publishing ([f8b08a0](https://github.com/AlexsJones/llmfit/commit/f8b08a0dd6f433bf9bdc7d93288ff653f263f03d))
* correct crates.io metadata and prepare for publishing ([2ce5858](https://github.com/AlexsJones/llmfit/commit/2ce5858ca4ae4514f95204c8b696d8173c5e9415)), closes [#58](https://github.com/AlexsJones/llmfit/issues/58)
* correct PCI ID reading on Linux and improve AMD GPU identification ([dcfde57](https://github.com/AlexsJones/llmfit/commit/dcfde579e5d3c3df6da3d2f7e152cf13e6a06fb8))
* correctly estimate VRAM for APU integrated GPUs ([2df3298](https://github.com/AlexsJones/llmfit/commit/2df329823597b057f3151ead64bbee38be393916))
* correctly estimate VRAM for APU integrated GPUs (Radeon Graphics) ([5797f10](https://github.com/AlexsJones/llmfit/commit/5797f10dbc43dafb0346be4d411c91b16025a5a2)), closes [#25](https://github.com/AlexsJones/llmfit/issues/25)
* Default theme uses terminal colors for light/dark compat ([72a6d8d](https://github.com/AlexsJones/llmfit/commit/72a6d8d7678ac90c7f3fafeb6a3cbd1ab25bb830)), closes [#67](https://github.com/AlexsJones/llmfit/issues/67)
* **desktop:** prevent XSS via inline onclick handler in modal ([#323](https://github.com/AlexsJones/llmfit/issues/323)) ([d20dbee](https://github.com/AlexsJones/llmfit/commit/d20dbeec9f8c3e7378aa2a386d684edda9432523))
* detect installed Ollama tag variants with suffixes ([#165](https://github.com/AlexsJones/llmfit/issues/165)) ([7092fa6](https://github.com/AlexsJones/llmfit/commit/7092fa63036a4cb04e5d0315665968f85268ac05))
* detect NVIDIA GB10/GB20 as unified memory GPUs ([a77cf36](https://github.com/AlexsJones/llmfit/commit/a77cf3631aa21339e215418cdeb9a8fab90891a1))
* detect NVIDIA GB10/GB20 as unified memory GPUs ([#83](https://github.com/AlexsJones/llmfit/issues/83), [#17](https://github.com/AlexsJones/llmfit/issues/17)) ([c0a584a](https://github.com/AlexsJones/llmfit/commit/c0a584a13d14bf71d03d12cc6f0bcd2c71f3d044))
* discover GGUF files in HuggingFace subdirectories ([#291](https://github.com/AlexsJones/llmfit/issues/291)) ([4e8a596](https://github.com/AlexsJones/llmfit/commit/4e8a5964d731f8ea39dcc25a474970b83f48b966))
* docker action version ([7a38a25](https://github.com/AlexsJones/llmfit/commit/7a38a25423d3841d3fec200815e55f1afc56e62d))
* docker action version pin ([da96503](https://github.com/AlexsJones/llmfit/commit/da965030a022c422232560259a053ec3d54ab620))
* **download:** fetch all shards of multi-part GGUF models ([b7494d9](https://github.com/AlexsJones/llmfit/commit/b7494d9910dff9e515337a6af9fc2acad59a2628))
* filter AWQ/GPTQ models by GPU compute capability ([e720a0e](https://github.com/AlexsJones/llmfit/commit/e720a0e9ebf43a1a2ec8630869c15dc9b827084e)), closes [#257](https://github.com/AlexsJones/llmfit/issues/257)
* find Tauri bundle in correct target directory ([19c0ac3](https://github.com/AlexsJones/llmfit/commit/19c0ac315f9376fbb9b2ff882720d6971c405951))
* **fmt:** align providers formatting and enrich TUI compare metrics ([a65e4c5](https://github.com/AlexsJones/llmfit/commit/a65e4c53a38f5a582a259801dcdc47ab7cf15f7a))
* hide MLX models on non-Apple-Silicon hardware ([#113](https://github.com/AlexsJones/llmfit/issues/113)) ([1e3cf8a](https://github.com/AlexsJones/llmfit/commit/1e3cf8a6a25f3f2e84a29a5becd01317e04a6e4b))
* hide MLX models on non-Apple-Silicon hardware ([#113](https://github.com/AlexsJones/llmfit/issues/113)) ([1052819](https://github.com/AlexsJones/llmfit/commit/1052819f40c8cc96efa1d46429f2e09a73c5e1fd))
* hide MLX-only models on non-Apple Silicon systems ([ab54fa9](https://github.com/AlexsJones/llmfit/commit/ab54fa927b79c0335e931a1989fa67b15eb1c8a3))
* ignore GGUF tests that require network access ([#306](https://github.com/AlexsJones/llmfit/issues/306)) ([8fd8f4d](https://github.com/AlexsJones/llmfit/commit/8fd8f4d9acc27506f78117029f3b540e6028ee6b))
* iGPU inflating GPU count and force-runtime being ignored ([#271](https://github.com/AlexsJones/llmfit/issues/271)) ([f584d7e](https://github.com/AlexsJones/llmfit/commit/f584d7edf7c42836441e3dcd0d5e911ca2bc4a2e))
* improve Android CPU and Vulkan GPU detection ([0ebb92a](https://github.com/AlexsJones/llmfit/commit/0ebb92a42960cdf4e5154616d5b1925d27f34a3b))
* improve download error messages for incompatible model formats ([0e2f9cc](https://github.com/AlexsJones/llmfit/commit/0e2f9ccc42a4ea3954e0320bce16d3500306fa46)), closes [#123](https://github.com/AlexsJones/llmfit/issues/123) [#198](https://github.com/AlexsJones/llmfit/issues/198)
* improve MoE tok/s estimate and clarify baseline speed labels ([81f88c4](https://github.com/AlexsJones/llmfit/commit/81f88c4ee78ab686fc47cefa72115025ca9e0154))
* incorrect installed model count in TUI status bar ([307c74f](https://github.com/AlexsJones/llmfit/commit/307c74f265b6e55f3c803eedf0a2e90bf8a499f3))
* invoke hf instead of huggingface-cli ([752680f](https://github.com/AlexsJones/llmfit/commit/752680f740540ae5f6e1c40a1f924fd6197a6634))
* invoke hf instead of huggingface-cli ([78f8268](https://github.com/AlexsJones/llmfit/commit/78f8268070f58c1c2647cfc0ceffa7ffab8a2e1e))
* pr review fixes ([25952da](https://github.com/AlexsJones/llmfit/commit/25952da6a6f707ad16fac33bdea0c17d38c5c064))
* prefer discrete GPU over integrated on Windows ([#303](https://github.com/AlexsJones/llmfit/issues/303)) ([3312a40](https://github.com/AlexsJones/llmfit/commit/3312a407901392806a623e653fa2b9bd20155b2a))
* prefer exact matches in info selection ([68272d4](https://github.com/AlexsJones/llmfit/commit/68272d40aeea33c482cc5e211902339ae6b72009))
* prefer exact matches in info selection ([883ea1f](https://github.com/AlexsJones/llmfit/commit/883ea1f90901f5020b655691588cdc70329c2adc))
* query local providers in `recommend` CLI command ([2477d1d](https://github.com/AlexsJones/llmfit/commit/2477d1dfa285a2eb3a46209c1f5487eaca20aeb8))
* regression in json only mode ([36e8503](https://github.com/AlexsJones/llmfit/commit/36e8503e6c21e40a11185fba0c84badd53f039a8))
* remove unused function to eliminate build-time warning ([4406582](https://github.com/AlexsJones/llmfit/commit/44065822115469d52bbd1afb5b8fe44f73c8756b))
* remove unused function to eliminate build-time warning ([289c8b8](https://github.com/AlexsJones/llmfit/commit/289c8b8fc6c9872f4c2ae20373e5c42c23ab0aa8))
* Runtimes are installed but no downloadable artifact exist ([9424204](https://github.com/AlexsJones/llmfit/commit/9424204e7364586889c0714b316433e3b5f63ea0))
* support owner-scoped MLX pulls and robust tag normalization ([7a9e476](https://github.com/AlexsJones/llmfit/commit/7a9e476e3b5b7926ccc7ee671aa538fffd5f05c8))
* support owner-scoped MLX repos and normalize MLX pull tags ([f0e516c](https://github.com/AlexsJones/llmfit/commit/f0e516c5c474228d86eb2bf86ff1caa1d5668f41))
* support sudo in piped install script ([9fa1960](https://github.com/AlexsJones/llmfit/commit/9fa19609064bb93585b2cf7a8af04ca07cb83de7))
* support sudo in piped install script ([cf61557](https://github.com/AlexsJones/llmfit/commit/cf615573acd329a6b4e525d4085b5cab301680f3)), closes [#158](https://github.com/AlexsJones/llmfit/issues/158)
* surface MoE offloaded RAM in JSON output ([96b73ff](https://github.com/AlexsJones/llmfit/commit/96b73ff7bc85b98ece85ea61f3c0d8acc5dd3559))
* surface MoE offloaded RAM in JSON output ([929eaad](https://github.com/AlexsJones/llmfit/commit/929eaad2aae7be9a8dd1fe962b5617fb10b0142e)), closes [#230](https://github.com/AlexsJones/llmfit/issues/230)
* **tui:** avoid borrow conflict when marking compare model ([ae60aa5](https://github.com/AlexsJones/llmfit/commit/ae60aa56ab0f4f9b039d309686c3d7ae714ea6f9))
* **tui:** write dashboard pid file under ~/.llmfit, not /tmp ([#324](https://github.com/AlexsJones/llmfit/issues/324)) ([00967f1](https://github.com/AlexsJones/llmfit/commit/00967f128d2606b84ba33e056c42ce78d265ef48))
* typo in CHANGELOG.md (suppor -&gt; support) ([1e26ec2](https://github.com/AlexsJones/llmfit/commit/1e26ec22f4d0bcbe071312c453fb01c5df8db926))
* typo in CHANGELOG.md (suppor -&gt; support) ([aa6cd00](https://github.com/AlexsJones/llmfit/commit/aa6cd00d8241b6391b575af408a1070db89dda66))
* update ([2b6767d](https://github.com/AlexsJones/llmfit/commit/2b6767dc15e8b40b52251c3a42e3f4a73694e2b0))
* update OpenClaw skill to match actual CLI output ([df585a7](https://github.com/AlexsJones/llmfit/commit/df585a759c8f24942f5ddfa2a15819cd7b35c540))
* update OpenClaw skill to match actual CLI output ([d53c192](https://github.com/AlexsJones/llmfit/commit/d53c1926a65f20baf9571e4ff73ae7d4207961fe))
* use accurate model counts instead of len()/2 for installed display ([7a273cc](https://github.com/AlexsJones/llmfit/commit/7a273cc3b42629106c818211d76ded1b2b06f316)), closes [#189](https://github.com/AlexsJones/llmfit/issues/189)
* use aggregated VRAM for multi-GPU fit scoring ([dc3db39](https://github.com/AlexsJones/llmfit/commit/dc3db39dafb62b7a41c045928ec0e08dd17ab66b))
* use aggregated VRAM for multi-GPU fit scoring ([f55882d](https://github.com/AlexsJones/llmfit/commit/f55882d9f664df98c6058fe7662cb95a20db35e9)), closes [#68](https://github.com/AlexsJones/llmfit/issues/68)
* use MoE active parameters for tok/s estimation and clarify baseline speed labels ([47e70a1](https://github.com/AlexsJones/llmfit/commit/47e70a17433fbd79b8a93be84a429b918dfd6661))
* use per-card VRAM instead of summed for multi-GPU systems ([2f1f9f4](https://github.com/AlexsJones/llmfit/commit/2f1f9f4e0ddac364e0a1ba4894adf9a6161926fc))
* use per-card VRAM instead of summed for multi-GPU systems ([f80b4b0](https://github.com/AlexsJones/llmfit/commit/f80b4b049173fa818019dc657b4bd8849d87e2d6))
* use quantization in path selection for both dense and MoE models ([#49](https://github.com/AlexsJones/llmfit/issues/49)) ([8703d43](https://github.com/AlexsJones/llmfit/commit/8703d431b971d95071cec40c66d514df7c995249))
* **web:** align dashboard hero with project tagline ([d4e77c1](https://github.com/AlexsJones/llmfit/commit/d4e77c167e835c9123b89c340b7c0d3579e2d26f))


### Performance Improvements

* **api:** optimize embedded asset serving and cache policy ([4d8c0cc](https://github.com/AlexsJones/llmfit/commit/4d8c0cc08384bcc1d3e50dc379014f8b3a31bf56))

## [0.9.6](https://github.com/AlexsJones/llmfit/compare/v0.9.5...v0.9.6) (2026-04-13)


### Features

* persist filter state across sessions ([#430](https://github.com/AlexsJones/llmfit/issues/430)) ([c3962e4](https://github.com/AlexsJones/llmfit/commit/c3962e4645ccd11bb6a5e5cf23a222e30c98af93))

## [0.9.5](https://github.com/AlexsJones/llmfit/compare/v0.9.4...v0.9.5) (2026-04-11)


### Features

* **plan:** KV cache fidelity + KV quant flag + TurboQuant gating ([1f1b6fc](https://github.com/AlexsJones/llmfit/commit/1f1b6fc25a04ee1cd7881f4e33becf7d24b95588))

## [0.9.4](https://github.com/AlexsJones/llmfit/compare/v0.9.3...v0.9.4) (2026-04-11)


### Bug Fixes

* **download:** fetch all shards of multi-part GGUF models ([b7494d9](https://github.com/AlexsJones/llmfit/commit/b7494d9910dff9e515337a6af9fc2acad59a2628))

## [0.9.3](https://github.com/AlexsJones/llmfit/compare/v0.9.2...v0.9.3) (2026-04-09)


### Features

* hardware simulation mode and CLI hardware overrides ([002eeff](https://github.com/AlexsJones/llmfit/commit/002eeff6acf61970c2d3bf471a3b2a1afc3f82ba)), closes [#322](https://github.com/AlexsJones/llmfit/issues/322)
* show approximate disk space usage ([#134](https://github.com/AlexsJones/llmfit/issues/134)) ([#304](https://github.com/AlexsJones/llmfit/issues/304)) ([32c835b](https://github.com/AlexsJones/llmfit/commit/32c835bf421a6b0727b32e5c2cddba8bf5c581e7))


### Bug Fixes

* **ci:** switch release-please to simple type for workspace version inheritance ([48be41a](https://github.com/AlexsJones/llmfit/commit/48be41aad22582a5921f0e1b3db2918aaad92950))
* **desktop:** prevent XSS via inline onclick handler in modal ([#323](https://github.com/AlexsJones/llmfit/issues/323)) ([d20dbee](https://github.com/AlexsJones/llmfit/commit/d20dbeec9f8c3e7378aa2a386d684edda9432523))
* **tui:** write dashboard pid file under ~/.llmfit, not /tmp ([#324](https://github.com/AlexsJones/llmfit/issues/324)) ([00967f1](https://github.com/AlexsJones/llmfit/commit/00967f128d2606b84ba33e056c42ce78d265ef48))

## [0.3.7](https://github.com/AlexsJones/llmfit/compare/v0.3.6...v0.3.7) (2026-02-21)


### Features

* add --memory flag to override GPU VRAM autodetection ([9a02f6e](https://github.com/AlexsJones/llmfit/commit/9a02f6e1616f59783ccff5b007c25213854f63b9))
* add --memory flag to override GPU VRAM autodetection ([39c5486](https://github.com/AlexsJones/llmfit/commit/39c5486aa3d94f9b9ef36e29642b64d848d0d2b0))
* add 15 popular models from HuggingFace ([128a020](https://github.com/AlexsJones/llmfit/commit/128a020323897a67ed5d12dd397bcf4924a6bf51))
* Add 15 popular models from HuggingFace (33→48 models) ([c45606b](https://github.com/AlexsJones/llmfit/commit/c45606bdb235b6bfe616bb616b1364a97e76f0c1))
* add homebrew tap support and update release workflow ([db09473](https://github.com/AlexsJones/llmfit/commit/db094734288d17a49d9c3c5c99859fe0d7dc976d))
* added arc support ([b5892fc](https://github.com/AlexsJones/llmfit/commit/b5892fc2ff313e71f57b7d793c7444d2aaadc0bd))
* added logo ([c21d416](https://github.com/AlexsJones/llmfit/commit/c21d4168f2bcd6da878848f9a6f97179d558606b))
* added moe ([ac7ffe4](https://github.com/AlexsJones/llmfit/commit/ac7ffe4ed79eb22ec43cf7bc20e8cd8d102d16a9))
* adding release please ([f2bfc7f](https://github.com/AlexsJones/llmfit/commit/f2bfc7fcf2587b74e05d8ad9d1041be6de456e69))
* append (WSL) to RAM label in tui when running under WSL ([e0397cf](https://github.com/AlexsJones/llmfit/commit/e0397cf51025b393b0d4024c4ae67200ee206390))
* caught some unavailable models on ollama ([b9f38da](https://github.com/AlexsJones/llmfit/commit/b9f38da9579040a7c2bada55838c5541474883ca))
* caught some unavailable models on ollama ([c0f7c20](https://github.com/AlexsJones/llmfit/commit/c0f7c20f61cdd9ae692de6ca66344befba2fafa9))
* detect installed Ollama models and support pulling from TUI ([4159aaf](https://github.com/AlexsJones/llmfit/commit/4159aaf304b3b421679f8231cf574465783d5b41))
* first pass ([855ad3d](https://github.com/AlexsJones/llmfit/commit/855ad3d34160cce6200c0ff128c34bcdcb0b922b))
* fixed up skill ([fcb712a](https://github.com/AlexsJones/llmfit/commit/fcb712a98ac785ad83ad689d5300f17cb80a3f1c))
* fixed up skill ([1f7d1de](https://github.com/AlexsJones/llmfit/commit/1f7d1de547a31202b9d34dd62bf543f5a22b2de7))
* fixing vram on apple bug ([5e08754](https://github.com/AlexsJones/llmfit/commit/5e087549c7c1523f4d5df72bd8a915330498a795))
* fixing vram on apple bug ([b3deca1](https://github.com/AlexsJones/llmfit/commit/b3deca1d9eac16283d0e9269c68a1af1dfc871ab))
* fixing vram on apple bug ([92ddb0e](https://github.com/AlexsJones/llmfit/commit/92ddb0e82579c6018d1acb4e3dfbe1df7d582605))
* fixing vram on apple bug ([42b2081](https://github.com/AlexsJones/llmfit/commit/42b2081577bed23176c0f87d1ad0b142cce23872))
* improvements based on [#12](https://github.com/AlexsJones/llmfit/issues/12) ([5428ef8](https://github.com/AlexsJones/llmfit/commit/5428ef8cdd42e88bced1459b55b480aab767637c))
* increased model count ([156b29d](https://github.com/AlexsJones/llmfit/commit/156b29deb077a1d66948254b370597a118fd5daf))
* increment version ([283bebb](https://github.com/AlexsJones/llmfit/commit/283bebb8eca5da2fc7124b665ae773fda48aed93))
* overall to the scoring system ([f475938](https://github.com/AlexsJones/llmfit/commit/f4759381d23b834e0a42a4699d23fb3f858fe677))
* overall to the scoring system ([b0696cf](https://github.com/AlexsJones/llmfit/commit/b0696cf297f1cb11247493355406d8b9c56510db))
* overall to the scoring system ([37e2e10](https://github.com/AlexsJones/llmfit/commit/37e2e10076f450f79165d92541baf04957ec2fe9))
* plumbing 2 ([1c615bb](https://github.com/AlexsJones/llmfit/commit/1c615bb57b7395f9be888245f8157dec2bab8bb4))
* plumbing 2 ([dd6a3ec](https://github.com/AlexsJones/llmfit/commit/dd6a3ec20e09ae72eada1fada73a6392c9673221))
* pull functionality ([923e7e7](https://github.com/AlexsJones/llmfit/commit/923e7e7463dd2bd53b6438ad3c8f2eb1f7a45af4))
* release plumbing ([7d21719](https://github.com/AlexsJones/llmfit/commit/7d217192bc1638f7ff69a22c2467d7d86da96641))
* release plumbing ([3accbb4](https://github.com/AlexsJones/llmfit/commit/3accbb42c99321fb6f8ade9d2f07af0fee93ed9e))
* reworked available models for download ([9adc84f](https://github.com/AlexsJones/llmfit/commit/9adc84f3041dca14fdcdc4437409b2b81eaca5a3))
* support for windows vulkan ([cc0fd61](https://github.com/AlexsJones/llmfit/commit/cc0fd619fa31e01c398c3c23f45aa915005670c8))
* supporting 94 models ([a652be3](https://github.com/AlexsJones/llmfit/commit/a652be31dd0cbe36f89572de7022e2a145fb3788))
* updated build actions ([1e65fdd](https://github.com/AlexsJones/llmfit/commit/1e65fddecb5f183870ddf1aa865dcaddba47523a))
* updated images ([9141109](https://github.com/AlexsJones/llmfit/commit/9141109f753ef38eb2b2eb5c604edb6ee0d7e371))
* updated models ([2d6c1d6](https://github.com/AlexsJones/llmfit/commit/2d6c1d66708186c0a21cb2f082a5b4e2fb03db90))
* updated tui to support multiple providers better and also multiple GPU support ([a3ca0bd](https://github.com/AlexsJones/llmfit/commit/a3ca0bd64647fa958c15bb7038a9e02df175fe67))
* updated urls ([f75ec27](https://github.com/AlexsJones/llmfit/commit/f75ec2750f325ff73725e5b8b194ba854c8579e9))
* updated version ([2cfc73e](https://github.com/AlexsJones/llmfit/commit/2cfc73ebdb6214f801e32880ff6451b2809bbb45))


### Bug Fixes

* correctly estimate VRAM for APU integrated GPUs ([72c8cb0](https://github.com/AlexsJones/llmfit/commit/72c8cb0e7873e0a8bcf4a10aee877bc38555299c))
* correctly estimate VRAM for APU integrated GPUs (Radeon Graphics) ([8da5c2a](https://github.com/AlexsJones/llmfit/commit/8da5c2a0443b73a3ac78ac087b0f08acdba6aaa9)), closes [#25](https://github.com/AlexsJones/llmfit/issues/25)
* update OpenClaw skill to match actual CLI output ([f38a0e5](https://github.com/AlexsJones/llmfit/commit/f38a0e56ef332bde8f3b03f8b06b5982fe90c1cc))
* update OpenClaw skill to match actual CLI output ([e1adbfd](https://github.com/AlexsJones/llmfit/commit/e1adbfd0abd786bc7a99496f20a7f81070bc8fe3))

## [0.3.6](https://github.com/AlexsJones/llmfit/compare/llmfit-v0.3.5...llmfit-v0.3.6) (2026-02-21)


### Features

* release plumbing ([7d21719](https://github.com/AlexsJones/llmfit/commit/7d217192bc1638f7ff69a22c2467d7d86da96641))
* release plumbing ([3accbb4](https://github.com/AlexsJones/llmfit/commit/3accbb42c99321fb6f8ade9d2f07af0fee93ed9e))

## [0.3.5](https://github.com/AlexsJones/llmfit/compare/llmfit-v0.3.4...llmfit-v0.3.5) (2026-02-21)


### Features

* add --memory flag to override GPU VRAM autodetection ([9a02f6e](https://github.com/AlexsJones/llmfit/commit/9a02f6e1616f59783ccff5b007c25213854f63b9))
* add --memory flag to override GPU VRAM autodetection ([39c5486](https://github.com/AlexsJones/llmfit/commit/39c5486aa3d94f9b9ef36e29642b64d848d0d2b0))
* add 15 popular models from HuggingFace ([128a020](https://github.com/AlexsJones/llmfit/commit/128a020323897a67ed5d12dd397bcf4924a6bf51))
* Add 15 popular models from HuggingFace (33→48 models) ([c45606b](https://github.com/AlexsJones/llmfit/commit/c45606bdb235b6bfe616bb616b1364a97e76f0c1))
* add homebrew tap support and update release workflow ([db09473](https://github.com/AlexsJones/llmfit/commit/db094734288d17a49d9c3c5c99859fe0d7dc976d))
* added arc support ([b5892fc](https://github.com/AlexsJones/llmfit/commit/b5892fc2ff313e71f57b7d793c7444d2aaadc0bd))
* added logo ([c21d416](https://github.com/AlexsJones/llmfit/commit/c21d4168f2bcd6da878848f9a6f97179d558606b))
* added moe ([ac7ffe4](https://github.com/AlexsJones/llmfit/commit/ac7ffe4ed79eb22ec43cf7bc20e8cd8d102d16a9))
* adding release please ([f2bfc7f](https://github.com/AlexsJones/llmfit/commit/f2bfc7fcf2587b74e05d8ad9d1041be6de456e69))
* append (WSL) to RAM label in tui when running under WSL ([e0397cf](https://github.com/AlexsJones/llmfit/commit/e0397cf51025b393b0d4024c4ae67200ee206390))
* caught some unavailable models on ollama ([b9f38da](https://github.com/AlexsJones/llmfit/commit/b9f38da9579040a7c2bada55838c5541474883ca))
* caught some unavailable models on ollama ([c0f7c20](https://github.com/AlexsJones/llmfit/commit/c0f7c20f61cdd9ae692de6ca66344befba2fafa9))
* detect installed Ollama models and support pulling from TUI ([4159aaf](https://github.com/AlexsJones/llmfit/commit/4159aaf304b3b421679f8231cf574465783d5b41))
* first pass ([855ad3d](https://github.com/AlexsJones/llmfit/commit/855ad3d34160cce6200c0ff128c34bcdcb0b922b))
* fixed up skill ([fcb712a](https://github.com/AlexsJones/llmfit/commit/fcb712a98ac785ad83ad689d5300f17cb80a3f1c))
* fixed up skill ([1f7d1de](https://github.com/AlexsJones/llmfit/commit/1f7d1de547a31202b9d34dd62bf543f5a22b2de7))
* fixing vram on apple bug ([5e08754](https://github.com/AlexsJones/llmfit/commit/5e087549c7c1523f4d5df72bd8a915330498a795))
* fixing vram on apple bug ([b3deca1](https://github.com/AlexsJones/llmfit/commit/b3deca1d9eac16283d0e9269c68a1af1dfc871ab))
* fixing vram on apple bug ([92ddb0e](https://github.com/AlexsJones/llmfit/commit/92ddb0e82579c6018d1acb4e3dfbe1df7d582605))
* fixing vram on apple bug ([42b2081](https://github.com/AlexsJones/llmfit/commit/42b2081577bed23176c0f87d1ad0b142cce23872))
* improvements based on [#12](https://github.com/AlexsJones/llmfit/issues/12) ([5428ef8](https://github.com/AlexsJones/llmfit/commit/5428ef8cdd42e88bced1459b55b480aab767637c))
* increased model count ([156b29d](https://github.com/AlexsJones/llmfit/commit/156b29deb077a1d66948254b370597a118fd5daf))
* increment version ([283bebb](https://github.com/AlexsJones/llmfit/commit/283bebb8eca5da2fc7124b665ae773fda48aed93))
* overall to the scoring system ([f475938](https://github.com/AlexsJones/llmfit/commit/f4759381d23b834e0a42a4699d23fb3f858fe677))
* overall to the scoring system ([b0696cf](https://github.com/AlexsJones/llmfit/commit/b0696cf297f1cb11247493355406d8b9c56510db))
* overall to the scoring system ([37e2e10](https://github.com/AlexsJones/llmfit/commit/37e2e10076f450f79165d92541baf04957ec2fe9))
* pull functionality ([923e7e7](https://github.com/AlexsJones/llmfit/commit/923e7e7463dd2bd53b6438ad3c8f2eb1f7a45af4))
* reworked available models for download ([9adc84f](https://github.com/AlexsJones/llmfit/commit/9adc84f3041dca14fdcdc4437409b2b81eaca5a3))
* support for windows vulkan ([cc0fd61](https://github.com/AlexsJones/llmfit/commit/cc0fd619fa31e01c398c3c23f45aa915005670c8))
* supporting 94 models ([a652be3](https://github.com/AlexsJones/llmfit/commit/a652be31dd0cbe36f89572de7022e2a145fb3788))
* updated build actions ([1e65fdd](https://github.com/AlexsJones/llmfit/commit/1e65fddecb5f183870ddf1aa865dcaddba47523a))
* updated images ([9141109](https://github.com/AlexsJones/llmfit/commit/9141109f753ef38eb2b2eb5c604edb6ee0d7e371))
* updated models ([2d6c1d6](https://github.com/AlexsJones/llmfit/commit/2d6c1d66708186c0a21cb2f082a5b4e2fb03db90))
* updated tui to support multiple providers better and also multiple GPU support ([a3ca0bd](https://github.com/AlexsJones/llmfit/commit/a3ca0bd64647fa958c15bb7038a9e02df175fe67))
* updated urls ([f75ec27](https://github.com/AlexsJones/llmfit/commit/f75ec2750f325ff73725e5b8b194ba854c8579e9))
* updated version ([2cfc73e](https://github.com/AlexsJones/llmfit/commit/2cfc73ebdb6214f801e32880ff6451b2809bbb45))


### Bug Fixes

* correctly estimate VRAM for APU integrated GPUs ([72c8cb0](https://github.com/AlexsJones/llmfit/commit/72c8cb0e7873e0a8bcf4a10aee877bc38555299c))
* correctly estimate VRAM for APU integrated GPUs (Radeon Graphics) ([8da5c2a](https://github.com/AlexsJones/llmfit/commit/8da5c2a0443b73a3ac78ac087b0f08acdba6aaa9)), closes [#25](https://github.com/AlexsJones/llmfit/issues/25)
* update OpenClaw skill to match actual CLI output ([f38a0e5](https://github.com/AlexsJones/llmfit/commit/f38a0e56ef332bde8f3b03f8b06b5982fe90c1cc))
* update OpenClaw skill to match actual CLI output ([e1adbfd](https://github.com/AlexsJones/llmfit/commit/e1adbfd0abd786bc7a99496f20a7f81070bc8fe3))
