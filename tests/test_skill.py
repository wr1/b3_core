from b3_core.skill import read_skill, skill_path


def test_skill_path_exists():
    path = skill_path()
    assert path.is_file()
    assert path.name == "SKILL.md"


def test_read_skill_has_frontmatter():
    text = read_skill()
    assert text.startswith("---\n")
    assert "name: b3-core" in text
    assert "homogenized properties for FEA" in text
