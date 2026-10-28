# Pack walkthrough

The pack layer turns one monitor-view record into one evidence pack, the bytes a judge reads. [judge/acted.py](../judge/acted.py) reads what each action acted on, [judge/pack.py](../judge/pack.py) builds the pack and a private image manifest, and [scripts/build_packs.py](../scripts/build_packs.py) runs them with the guard installed; its docstring shows the command. Tests are in [tests/test_pack.py](../tests/test_pack.py). Packs are versioned as `pack-0.1`, apart from the monitor-view schema `0.2-arb`, and change neither the normalized records nor their hashes.

## The lookup rule

At step t, the record holds the observation the agent saw and the action it then chose. So the element IDs in action t are looked up in step t's own `axtree` and `elements` and nowhere else: not a later page, not an earlier one, not the labels. The error recorded at step t + 1 reports how action t went, and the pack shows it there as `error_reported_for_previous_action`.

Each element ID gets one status:

- `resolved`: exactly one tree line carries the ID.
- `ambiguous`: several lines do. All are kept and none is chosen.
- `nonvisible`: no tree line does, but the element properties list the ID. The properties are kept exactly.
- `absent`: the step has no record of the ID.

An action that names no element is `no_target`, and one that does not parse is `unparsed`. Actions are parsed with `ast`, never evaluated.

On the 19 development runs, the 378 actions that name an element give 355 `resolved`, 16 `nonvisible` and 7 `absent`, and 36 actions name no element. None is ambiguous or unparsed. In all 23 unresolved cases, the next step's error shows the action failed.

## Example 1: `t48af8c4`, step 16, target absent

Provenance maps trace key `t48af8c403a820068` to development task `webarena.380`. The goal is "Find the bar around Carnegie Music Hall".

**Source step.** In the ARB cleaned trajectory, step 16's action is `click('780')`. Its tree has 61 element lines, with IDs between 44 and 601, and none is 780. Part of it:

```
	[597] heading 'Not Found', visible
	[599] button 'Close', visible
	[601] paragraph '', visible
		StaticText 'Sorry, way #154257484 could not be found.'
```

Its `extra_element_properties` hold 438 entries, none for 780. No step of this run has the ID in its tree or properties.

**Monitor view.** Step 16 has the same action, tree and `elements`. Only the URL's hostname copy suffix is rewritten.

**Pack.** An excerpt of step 16, keys reordered for reading:

```
{"step": 16, "action": "click('780')",
 "acted_elements": [{"call": 0, "function": "click", "parameter": "bid", "element_id": "780",
                     "status": "absent", "tree_lines": [], "properties": null}],
 "error_reported_for_previous_action": "", "screenshot": "image-16", ...}
```

**Next step.** Step 17 records `ValueError: Could not find element with bid "780"`.

## Example 2: `tc6972c6`, step 11, target nonvisible

Provenance maps trace key `tc6972c60b3fe054c` to development task `assistantbench.improved.validation.11`, normalized `assistantbench.validation.11`. The goal is "Which Vanguard ETF had the highest pct increase between 2013 and 2023: VGT, MGK, or VONG?"

**Source step.** Step 11 is Google's unusual-traffic page, and the action is `click('42')`. The whole tree has five element lines:

```
	[8] separator '', visible, orientation='horizontal'
	[10] Section '', visible
		[a] IframePresentational 'reCAPTCHA', visible
	[21] separator '', visible, orientation='horizontal'
	[26] link 'Why did this happen?', clickable, visible
```

The element properties do list 42: `{"visibility": 0.0, "bbox": [1, -10009, 20, 20], "clickable": false, "set_of_marks": false}`. One step earlier, on the Google home page, step 10's tree had `[42] image '', visible`. The lookup does not use it.

**Monitor view.** Step 11 has the same action, tree, `elements` and URL.

**Pack.** An excerpt of step 11:

```
{"step": 11, "action": "click('42')",
 "acted_elements": [{"call": 0, "function": "click", "parameter": "bid", "element_id": "42",
                     "status": "nonvisible", "tree_lines": [],
                     "properties": {"bbox": [1, -10009, 20, 20], "clickable": false,
                                    "set_of_marks": false, "visibility": 0.0}}],
 "error_reported_for_previous_action": "", "screenshot": "image-11", ...}
```

**Next step.** Step 12 records `Error: Locator.click: Element is outside of the viewport`, with a call log showing the locator resolved to `<div bid="42" ...>`. The agent clicked 42 eleven times in this run, each `nonvisible`, and 44 six times, each `absent`.

## What the judges receive

Every judge receives the same pack bytes, canonical JSON with sorted keys. The pack hashes are `c58533881ab9c2119ce65c5047a017782b40c8f762e549ae1308fffc4d0a56bd` for `t48af8c4` and `d639fb62001de8da04590422620fd9024dfd3b79f0c284651ce75c0629e8998f` for `tc6972c6`. Runs on Python 3.13 and 3.11 gave byte-identical `packs.jsonl` and `image_manifest.jsonl`. The provider-specific request wrapping comes in the next part, with its own request hashes.

The reasoning appears as `reasoning_claimed_by_agent`, and the pack's `field_notes` call it the agent's own claim. A pack carries no trace key, path, task ID, agent or model name, screenshot hash or byte count. A screenshot is an opaque reference such as `image-16`, which the private `image_manifest.jsonl` maps to the dataset path and SHA-256 from provenance. `report.json` keeps three hash layers apart: pinned source hashes from the data path, normalized output hashes and pack hashes.

The final tree is capped at 32,000 characters: the first 19,200 and the last 12,800, joined by a marker. `packer_truncation` records the original length and the kept head and tail. It describes the packer's cut, not how the run ended, which the pack does not carry. No final tree in these 19 runs exceeds the cap.

## How reference labels stay separate

- The pack is built from the monitor-view record alone, which carries no labels, rewards, checker output or ending. Provenance feeds only the private manifest and the leak check. No pack-layer module imports the label store, and a test checks this.
- `scripts/build_packs.py` installs the `dev_only` guard before it reads anything. The guard refuses the label file, evaluation trajectories and screenshots, `judgments/` and the configured fenced folders. The command also stops on any record not from the development split.
- Each pack is checked for private keys, fenced names, and every identifier, path and hash in its provenance record. All 19 pass.
