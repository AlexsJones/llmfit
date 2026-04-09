# Changelog

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
