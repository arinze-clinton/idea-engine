# Idea engine

A Claude Code skill for finding genuinely new ideas, approaches and solutions to a question, instead of a blend of the top three search results.

It works the way an "idea plotter" does: collect every existing answer to your question from many fields, sort them into families, find the empty spaces between them, and fill those with new ideas. Blind critics then judge each idea for how new it really is in the world and how useful it is. You get proven answers and new ones side by side, on a page where you can pick.

## What you need

- [Claude Code](https://claude.com/claude-code)
- [uv](https://docs.astral.sh/uv/) for the small Python helper script (it installs its own packages)
- About 0.6 GB of free space: the first run downloads a small local search model

Optional: a screen-library connector such as Mobbin, and a built-in browser if you want collectors to browse your own inspiration sites.

## Install

Clone this repo and copy the `idea-engine` folder into your Claude Code skills folder:

```bash
git clone https://github.com/arinze-clinton/idea-engine.git
mkdir -p ~/.claude/skills && cp -R idea-engine/idea-engine ~/.claude/skills/
```

Start a new Claude Code session so it picks the skill up.

## Use

Ask in your own words. Any of these start it:

- "idea engine: how could our onboarding handle an empty workspace?"
- "fresh ideas for our pricing page"
- "map everything that exists for in-app search with no results"
- "what hasn't been done in habit tracking?"

It grills you first with a few rounds of questions, each with a suggested answer, so the run has full context. It also tells you up front if your question is one where proven answers beat new ones.

A full run is big. It starts many helper agents, takes about an hour and uses a lot of tokens, so save it for questions that deserve it.

## What's inside

- `SKILL.md`: the full method, step by step
- `scripts/engine.py`: the clerical helper (gathering cards, merging repeats, the local search box, joining critic verdicts, drawing the pages). It enforces the method's rules, such as no ideas for gaps that failed their check, decoys that catch fooled critics, and a minimum number of searches per verdict.
- `scripts/picks.html` and `scripts/page.html`: templates for the picks page and the map
