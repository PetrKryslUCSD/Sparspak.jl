#=
This julia script converts fortran 90 code into julia.
It uses naive regex replacements to do as much as possible,
but the output WILL need further cleanup.
Known conversion problems such as GOTO are commented and marked with FIXME
Most variable declaration lines are entirely deleted, which may or
may not be useful. 
To run from a shell: 
julia fortran-julia.jl filename.f90
Output is written to filename.jl.
=#

import Pkg; Pkg.add("DataStructures")
using DataStructures

function doit()
    stem, ext = splitext(filename)
    outfile = stem * "-auto-translated.jl"

    # Regex/substitution pairs for replace(). Order matters here.
    replacements = OrderedDict(
    # Lowercase everything not commented
    r"^(?!.*!).*"m => lowercase,
    # Lowercase start of lines with comments
    r"^.*!"m => lowercase,
    # Remove '&' multiline continuations
    r"\s*&\s*" => "",
    # Comments use # not !
    "!" => "#",
    # Powers use ^ not **
    "**" => "^",
    # Only double quotes allowed for strings
    "'" => "\"",
    # DO loop to for loop
    r"do (.*),(.*)" => s"for \1:\2",
    # Spaces around math operators
    r"([\*\+\/=])(?=\S)" => s"\1 ",
    r"(?<=\S)([\*\+\/=])" => s" \1",
    # Spaces around - operators, except after e
    # r"([^e][\-])(\S)" => s"\1 \2",
    r"(?<!\W\de)(\h*\-\h*)" => s" - ",
    r"(= =)" => "==",
    # Space after all commas
    r"(,)(\S)" => s"\1 \2",
    # Replace space after all (
    r"(\()(\s*)" => "(",
    # Replace space before all (
    r"(\s*)(\))" => ")",
    # Replace ELSEIF/ELSE IF with elseif 
    r"(\s+)else\s*if" => s"\1elseif",
    # Replace IF followed by ( to if (
    r"(\s+)(elseif|if)\(" => s"\1\2 (",
    # Add end after single line if with an = assignment
    # r"if\s*\((.*?)\)(\s*.*\s*.*=\s*.*)(\n)" => s"if (\1) \2 end\3",
    # Remove THEN
    r"([)\s])then(\s+)" => s"\1\2",
    # Replace END XXXX with end
    r"(\s+)end\s*do" => s"\1end",
    # Replace END XXXX with end
    r"(\s+)end\s*if" => s"\1end",
    # Replace END FUNCTION with end
    r"end function .*" => s"end",
    # Replace END subroutine with end
    r"end subroutine .*" => s"end",
    # Replace allocate
    r"(allocate\((.*)\((.*)\)\))" => s"FIXME \1",
    # Replace exponent function
    # r"(\W)exp\(" => s"\1exp(",
    # Reorganise functions and doc strings. This may be very project specific.
    r"#\^\^+\s*subroutine\s*(\w+)([^)]+\))\s*(.*?)#\^\^\^+"sm => 
    Base.SubstitutionString("\"\"\"\n\\3\"\"\"\nfunction \\1\\2"),
    r"real \(double\) function\s*(\w+)([^)]+\))\s*(.*?)\#\^\^\^+"sm => 
    Base.SubstitutionString("\"\"\"\n\\3\"\"\"\nfunction \\1\\2"),
    # Don't need CALL
    r"(\s*)call(\h+)" => s"\1",
    # Use real math symbols
    #   "gamma" => "Γ",
    #   "theta" => "Θ",
    #   "epsilon" => "ϵ",
    #   "lambda" => "λ",
    #   "alpha" => "α",
    # Swap logical symbols
    ".true." => "true",
    ".false." => "false",
    r"\s*\.or\.\s*" => " || ",
    r"\s*\.and\.\s*" => " && ",
    r"\s*\.not\.\s*" => " ! ",
    r"\s*\.eq\.\s*" => " == ",
    r"\s*\.ne\.\s*" => " != ",
    r"\s*\.le\.\s*" => " <= ",
    r"\s*\.ge\.\s*" => " >= ",
    r"\s*\.gt\.\s*" => " > ",
    r"\s*\.lt\.\s*" => " < ",
    # Remove (expression) brackets after if
    # r"if \((.*)\)(\s*\n)" => s"if \1\2",
    # Format floats as "5.0" not "5."
    r"(\W\d+)\.(\D)" => s"\1.0\2",
    # Tab to 4 spaces
    r"\t" => "    ",
    # Replace component access
    r"%" => ".",
    # Replace doubled !
    r"## " => "#",
    # Replace #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    r"#\^*" => "#",
    # # Replace opening and closing of comments
    r"##>" => "# \"\"\"",
    r"##<(\s*.*)" => Base.SubstitutionString("# \\1\\n# \"\"\""),
    r"# \"\"\" Purpose:" => "# \"\"\"",
    # r"##" => "",
    # Replace suberror with error and mark for fixup
    r"(\W)suberror\((.*?),.*?\)" => s"\1 error(\2)",
    # Mark #FIXME the various things this script can't handle
    r"(write|goto|while\s)" => s"FIXME \1",
    )

    # Patterns to remove
    removal = [
    #   # Trailing whitespace
      r"\h*$"m,
    #   # Variable declarations
    #   r"\n\s*real\s.*",
    #   r"\n\s*real, external\s.*",
    #   r"\n\s*integer\s.*",
      r"\n\s*implicit none",
    #   r"\n\s*logical\s.*",
    #   # Import statements
      r"\n\s*use\s.*",
      r"\n\s*contains\s.*",
    ]

    # Load the file from the first command line argument

    code = read(filename,String)

    # Process replacements and removals.
    for (f, r) in replacements
        code = replace(code, f => r)
    end
    for r in removal
        code = replace(code, r => "")
    end
    # println(code)


    # Write the output to a .jl file with the same filename stem.
    
    open(outfile,"w") do f
        write(f, code)
    end

    return nothing    
end

    filename = ARGS[1]
    doit()