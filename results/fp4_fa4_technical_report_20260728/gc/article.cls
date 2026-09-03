%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Graphcore article class, built by adapting:
% ---------------------------------------------------------
% Tau
% LaTeX Class
% Version 2.4.4 (28/02/2025)
%
% Author:
% Guillermo Jimenez (memo.notess1@gmail.com)
%
% License:
% Creative Commons CC BY 4.0
% --------------------------------------------------------
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%----------------------------------------------------------
% CLASS CONFIGURATION
%----------------------------------------------------------

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{gc/article}[2025/10/01 Graphcore article LaTeX class]
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{extarticle}}
\ProcessOptions\relax
\LoadClass{extarticle}
\AtEndOfClass{\RequirePackage{microtype}}

%----------------------------------------------------------
% REQUIRED PACKAGES
%----------------------------------------------------------

\RequirePackage[T1]{fontenc}
\RequirePackage{etoolbox}
\RequirePackage[framemethod=tikz]{mdframed}
\RequirePackage{titling}
\RequirePackage[switch]{lineno}
\RequirePackage[bottom,hang,flushmargin,ragged]{footmisc}
\RequirePackage{fancyhdr}
\RequirePackage{xifthen}
\RequirePackage{adjustbox}
\RequirePackage{lastpage}
\RequirePackage[explicit]{titlesec}
\RequirePackage{booktabs}
\RequirePackage{array}
\RequirePackage{caption}
\RequirePackage{setspace}
\RequirePackage{iflang}
\RequirePackage{listings}
\RequirePackage{lipsum}
\RequirePackage{supertabular}
\RequirePackage{listings}
\RequirePackage{csquotes}
\RequirePackage{ragged2e}
\RequirePackage{subcaption}
\RequirePackage{stfloats}

%----------------------------------------------------------
% TAU CUSTOM PACKAGES (location/name)
%----------------------------------------------------------

\RequirePackage{gc/article-envs}

%----------------------------------------------------------
% PACKAGES FOR FIGURES
%----------------------------------------------------------

\usepackage{graphicx}
\RequirePackage{here}
% \graphicspath{{figures/}{./}}

%----------------------------------------------------------
% PACKAGES FOR TABLES
%----------------------------------------------------------

\usepackage{adjustbox}
\RequirePackage{colortbl}
\RequirePackage{tcolorbox}

%----------------------------------------------------------
% GEOMETRY PACKAGE SETUP
%----------------------------------------------------------

\RequirePackage[
    \if@twocolumn
        left=1.5cm,
        right=1.5cm,
    \else
        left=2.5cm,
        right=2.5cm,
    \fi
    top=2.5cm,
    bottom=2.5cm,
    headsep=0.75cm
]{geometry}

%----------------------------------------------------------
% MATH PACKAGES
%----------------------------------------------------------

\RequirePackage{amsmath}
\RequirePackage{amsfonts}
\RequirePackage{mathtools}
\RequirePackage{amssymb}

% Equation skip value
\newlength{\eqskip}\setlength{\eqskip}{8pt}
\expandafter\def\expandafter\normalsize\expandafter{%
    \normalsize%
    \setlength\abovedisplayskip{\eqskip}%
    \setlength\belowdisplayskip{\eqskip}%
    \setlength\abovedisplayshortskip{\eqskip-\baselineskip}%
    \setlength\belowdisplayshortskip{\eqskip}%
}

%----------------------------------------------------------
% FONTS
%----------------------------------------------------------

\usepackage[quiet]{fontspec}
\usepackage[warnings-off={not-ot-math,mathtools-overbracket,mathtools-colon}]{unicode-math}
\setmathfont{Latin Modern Math}
\setmainfont{Latin Modern Roman}
\setsansfont{Latin Modern Sans}
\newfontface\titleface{Latin Modern Sans}
\renewcommand{\ttdefault}{lmtt}
% https://tex.stackexchange.com/questions/353443/xetex-unicode-math-tilde-on-bold-characters
\renewcommand{\mathbf}[1]{\symbfit{#1}}


%----------------------------------------------------------
% URLs STYLE
%----------------------------------------------------------

\RequirePackage{url}
\RequirePackage{xurl}
\renewcommand\UrlFont{\selectfont}

%----------------------------------------------------------

\RequirePackage[colorlinks=true,allcolors=gcdarkgrey]{hyperref}
\RequirePackage{cleveref}
\RequirePackage{bookmark}

%----------------------------------------------------------
% ITEMS
%----------------------------------------------------------

\RequirePackage{enumitem}
\setlist{noitemsep}

%----------------------------------------------------------
% PREDEFINED LENGTHS
%----------------------------------------------------------

\setlength{\columnsep}{30pt}

%----------------------------------------------------------
% FIRST PAGE, HEADER AND FOOTER
%----------------------------------------------------------

% New commands
\newcommand{\footerfont}{\normalfont\sffamily\mathversion{graphikmath-regular}\fontsize{7}{9}\selectfont}
\newcommand{\institution}[1]{\def\@institution{#1}}
\newcommand{\footinfo}[1]{\def\@footinfo{#1}}
\newcommand{\leadauthor}[1]{\def\@leadauthor{#1}}
\newcommand{\permissions}[1]{\def\@permissions{#1}}
\newcommand{\theday}[1]{\def\@theday{#1}}
\pagestyle{fancy}

% Number style
\pagenumbering{arabic} % Roman

% First page style
\fancypagestyle{firststyle}{
    \fancyfoot[R]{
        {\ifx\@institution\undefined\else\footerfont\@institution\hspace{10pt}\fi}
        {\ifx\@theday\undefined\else\footerfont\bfseries\@theday\hspace{10pt}\fi}
        {\ifx\@footinfo\undefined\else\footerfont\@footinfo\hspace{10pt}\fi}
        \footerfont\textbf{\thepage\textendash\pageref{LastPage}}
    }
    \fancyfoot[L]{\ifx\@permissions\undefined\else\footerfont\@permissions\fi}
    \fancyhead[C]{}
    \fancyhead[R]{}
    \fancyhead[L]{}
}

% Fancy head
\fancyhead[C]{}
\fancyhead[RE]{\footerfont\@title}
\fancyhead[RO]{\footerfont\ifx\@leadauthor\undefined\else\@leadauthor\fi}
\fancyhead[LO]{\footerfont\@title}
\fancyhead[LE]{\footerfont\ifx\@leadauthor\undefined\else\@leadauthor\fi}

% Fancy foot
\fancyfoot[C]{}
\fancyfoot[LE]{\footerfont\textbf{\thepage}\hspace{10pt}\ifx\@permissions\undefined\else\@permissions\fi}
\fancyfoot[RO]{\footerfont\ifx\@permissions\undefined\else\@permissions\hspace{10pt}\fi\textbf{\thepage}}
\fancyfoot[RE,LO]{\raisebox{-6pt}{\includegraphics[width=1cm]{gc/gcsymbol.png}}}

% Head and foot rule
\renewcommand{\headrulewidth}{0pt}      % No header rule
\renewcommand{\footrulewidth}{0pt}      % No footer rule

%----------------------------------------------------------
% ABSTRACT STYLE
%----------------------------------------------------------

% Abstract font
\newcommand{\absfont}{\selectfont\small\sffamily\bfseries}
% Abstract text font
\newcommand{\abstextfont}{\selectfont\sffamily\small}

% Keywords new command
\newcommand{\keywords}[1]{\def\@keywords{#1}}
% Keyword font
\newcommand{\keywordsfont}{\selectfont\small\sffamily\bfseries}
% Keyword text font
\newcommand{\keywordsfonttext}{\selectfont\sffamily\small}

\def\xabstract{abstract}
\long\def\abstract#1\end#2{\def\two{#2}\ifx\two\xabstract
    \long\gdef\theabstract{\ignorespaces#1}
    \def\go{\end{abstract}}
\else
    #1\end{#2}
    \gdef\theabstract{\vskip12pt
    \vskip12pt}
    \let\go\relax\fi
\go}

\newcommand{\gcabstract}{
    \begin{gcabstractenv}
    \noindent
    \parbox{\dimexpr\linewidth}{
	\vskip3pt{\absfont\abstractname}\\[3pt]
    \abstextfont\theabstract
    }
    \@ifundefined{@keywords}{\vskip15pt}{%
        \vskip6pt
        \noindent
        \parbox{\dimexpr\linewidth}{
        {
            {\keywordsfont Keywords\ignorespaces} --- {\keywordsfonttext\@keywords}
        }
        }
        \vskip12pt
    }
    \end{gcabstractenv}
}

%----------------------------------------------------------
% TITLE STYLE
%----------------------------------------------------------

% New commands
\newcommand{\journalname}[1]{\def\@journalname{#1}}
\newcommand{\titlepos}{\raggedright}
\newcommand{\titlefont}{\titleface\color{gcpink}\fontsize{18}{22}\mathversion{graphikmath}\selectfont\MakeUppercase}
\newcommand{\authorfont}{\normalfont\sffamily\bfseries\fontsize{9}{11}}
\newcommand{\teamfont}{\sffamily\fontsize{9}{11}\selectfont}

\renewcommand{\@maketitle}{
    {\ifx\@journalname\undefined\vskip0pt\else\vskip-7pt\RaggedRight\sffamily\bfseries\fontsize{8}{14}\@journalname\par\fi}
    {\titlepos\titlefont\@title\par\vskip8pt}
    {\authorfont\titlepos\@author\par\vskip8pt}
    {\ifx\@institution\undefined\vskip14pt\else\teamfont\titlepos\@institution\par%
    \if@twocolumn\vskip22pt\else\vskip3pt\fi
    \fi}
}

%----------------------------------------------------------
% SECTION STYLE
%----------------------------------------------------------

\setcounter{secnumdepth}{5}

\titleformat{\section}
    {\color{gcpink}\sffamily\large\bfseries}
    {\thesection.}
    {0.5em}
    {#1}
    []

\titleformat{name=\section,numberless}
    {\color{gcpink}\sffamily\large\bfseries}
    {}
    {0em}
    {#1}
    []

\titleformat{\subsection}[block] % You may change it to "runin"
    {\bfseries\sffamily}
    {\thesubsection.}
    {0.5em}
    {#1}      % If using runin, change #1 to '#1. ' (space at the end)
    []

\titleformat{\subsubsection}[block] % You may change it to "runin"
    {\small\bfseries\sffamily}
    {\thesubsubsection.}
    {0.5em}
    {#1}      % If using runin, change #1 to '#1. ' (space at the end)
    []

\titleformat{\paragraph}[runin]
    {\bfseries}
    {}
    {0em}
    {#1}

\titleformat{\subparagraph}[runin]
    {\itshape}
    {}
    {0em}
    {#1}

\titlespacing*{\section}{0pc}{3ex \@plus4pt \@minus3pt}{5pt}
\titlespacing*{\subsection}{0pc}{2.5ex \@plus3pt \@minus2pt}{2pt}
\titlespacing*{\subsubsection}{0pc}{2ex \@plus2.5pt \@minus1.5pt}{2pt}
\titlespacing*{\paragraph}{0pc}{1.5ex \@plus2pt \@minus1pt}{12pt}
\titlespacing*{\subparagraph}{0pc}{1.5ex \@plus2pt \@minus1pt}{12pt}

%----------------------------------------------------------
% TABLE OF CONTENTS
%----------------------------------------------------------

\newlength{\tocsep}
\setlength\tocsep{1.5pc} % Sets the indentation of the sections in the table of contents
\setcounter{tocdepth}{5} % Three levels in the table of contents section: sections, subsections and subsubsections

\RequirePackage{titletoc}
\contentsmargin{0cm}

\titlecontents{section}[\tocsep]
    {\addvspace{4pt}\sffamily\selectfont\bfseries}
    {\contentslabel[\thecontentslabel]{\tocsep}}
    {}
    {\hfill\thecontentspage}
    []

\titlecontents{subsection}[3pc]
    {\addvspace{4pt}\small\sffamily\selectfont}
    {\contentslabel[\thecontentslabel]{\tocsep}}
    {}
    {\ \titlerule*[.5pc]{.}\ \thecontentspage}
    []

\titlecontents*{subsubsection}[3pc]
    {\footnotesize\sffamily\selectfont}
    {}
    {}
    {}
    [\ \textbullet\ ]

%----------------------------------------------------------
% FOOTNOTE STYLE
%----------------------------------------------------------

\definecolor{black50}{gray}{0.5}
\renewcommand*{\footnotelayout}{\normalfont\fontsize{6}{8}}
\renewcommand{\footnoterule}{
    \kern -3pt
    {\color{black50} \hrule width 72pt height 0.25pt}
    \kern 2.5pt
}

%----------------------------------------------------------
% FIGURE-, TABLE-, LISTINGS- CAPTION STYLE
%----------------------------------------------------------

% Figure
\captionsetup[figure]{format=plain, labelsep=period, textfont={small}, justification=centering, labelfont={small,bf,sf}}

% Table
\captionsetup*[table]{labelfont={small,bf,sf},textfont={small},skip=10pt,position=below,labelsep=period} % Change it to "above" if you prefer the caption above the table.
\newcommand{\tabletext}[1]{{\setlength{\leftskip}{9pt}\fontsize{7}{9}\vskip2pt\itshape\selectfont#1}}

\captionsetup[lstlisting]{font=small, labelfont={bf,sf}, belowskip=3pt, position=below, textfont=small, labelsep=period}
% \renewcommand\lstlistingname{Code}
% \renewcommand\lstlistlistingname{Code}

%----------------------------------------------------------
% LISTINGS STYLE
%----------------------------------------------------------

% Defined colors for listings environment
\definecolor{taucodeback}{RGB}{248, 248, 248}
\definecolor{taucodecomment}{RGB}{1, 136, 0}
\definecolor{taucodekey}{RGB}{53, 53, 128}
\definecolor{taucodestring}{RGB}{122, 36, 47}
\definecolor{taugray}{RGB}{0.5,0.5,0.5}
\definecolor{tauredmatlab}{RGB}{199, 78, 0}
\definecolor{taublue}{RGB}{43, 43, 255}

% General
\lstdefinestyle{taucoding}{
    backgroundcolor=\color{taucodeback},
    commentstyle=\color{taucodecomment},
    keywordstyle=\bfseries\color{taucodekey},
    numberstyle=\tiny\color{taugray},
    stringstyle=\color{taucodestring},
    basicstyle=\footnotesize\ttfamily,
    breakatwhitespace=false,
    basicstyle=\small\ttfamily,
    breaklines=true,
    captionpos=b,
    keepspaces=true,
    numbers=left,				 	% if "none" change the values
    numbersep=8pt,    				% 0pt
    showspaces=false,
    showstringspaces=false,
    showtabs=false,
    tabsize=2,
    aboveskip=12pt,
    belowskip=8pt,
    xleftmargin=12pt,				% 0pt
    xrightmargin=3pt,
    rulecolor=\color{taugray},
    frame=single,
    columns=fullflexible,
    postbreak=\mbox{\small\textcolor{taugray}{$\hookrightarrow$}\space},
    literate=
        {0}{{\textcolor{taublue}{0}}}{1}
        {1}{{\textcolor{taublue}{1}}}{1}
        {2}{{\textcolor{taublue}{2}}}{1}
        {3}{{\textcolor{taublue}{3}}}{1}
        {4}{{\textcolor{taublue}{4}}}{1}
        {5}{{\textcolor{taublue}{5}}}{1}
        {6}{{\textcolor{taublue}{6}}}{1}
        {7}{{\textcolor{taublue}{7}}}{1}
        {8}{{\textcolor{taublue}{8}}}{1}
        {9}{{\textcolor{taublue}{9}}}{1}
}

\lstset{style=taucoding}

%----------------------------------------------------------
% BIBLATEX
%----------------------------------------------------------

\RequirePackage[
    backend=biber,
    style=ieee,
    sorting=ynt
]{biblatex}

%----------------------------------------------------------

\endinput
