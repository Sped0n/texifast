from texifast.helpers import refine_math_block


def test_refine_math_block() -> None:
    text = "This is a math block $$x^2$$ and this is another math block $$y^2$$"
    assert (
        refine_math_block(text)
        == "This is a math block\n\n$$\nx^2\n$$\n\nand this is another math block$$\ny^2\n$$"
    )
