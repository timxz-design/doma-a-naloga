import os
import streamlit as st
import chromadb
import pandas as pd
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# ─── Constants ────────────────────────────────────────────────────────────────

CHROMA_PATH     = "./chroma_db"
COLLECTION_NAME = "startup_rag"
MODEL_NAME      = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE      = 300
CHUNK_OVERLAP   = 50

# ─── CSS ──────────────────────────────────────────────────────────────────────

def inject_css():
    st.markdown("""
    <style>
    :root {
        --navy:  #0d1b2a;
        --gold:  #c9a84c;
        --gold2: #e8c97a;
        --light: #e8e0d0;
        --muted: #8a9ab5;
        --card:  #16263b;
        --card2: #1e3350;
        --border: #2a4a6b;
    }

    /* App background */
    [data-testid="stApp"] {
        background-color: var(--navy);
        color: var(--light);
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: var(--card);
        border-right: 1px solid var(--border);
    }
    [data-testid="stSidebar"] * {
        color: var(--light) !important;
    }

    /* Headings */
    h1 { color: var(--gold) !important; font-size: 2rem !important; letter-spacing: -0.5px; }
    h2 { color: var(--gold) !important; font-size: 1.4rem !important; }
    h3 { color: var(--gold2) !important; font-size: 1.1rem !important; }

    /* Buttons */
    .stButton > button {
        background-color: var(--card2);
        color: var(--gold);
        border: 1px solid var(--gold);
        border-radius: 6px;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        background-color: var(--gold);
        color: var(--navy);
    }

    /* Text inputs */
    .stTextInput input {
        background-color: var(--card);
        color: var(--light);
        border: 1px solid var(--border);
        border-radius: 6px;
    }
    .stTextInput input:focus {
        border-color: var(--gold);
        box-shadow: 0 0 0 2px rgba(201,168,76,0.2);
    }

    /* Slider */
    .stSlider [data-baseweb="slider"] {
        color: var(--gold);
    }

    /* Metrics */
    [data-testid="stMetric"] {
        background-color: var(--card);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 1rem 1.2rem;
    }
    [data-testid="stMetricLabel"] { color: var(--muted) !important; }
    [data-testid="stMetricValue"] { color: var(--gold) !important; }

    /* Dataframe */
    [data-testid="stDataFrame"] {
        border: 1px solid var(--border);
        border-radius: 8px;
    }

    /* General text */
    p, li { color: var(--light); line-height: 1.7; }
    label { color: var(--muted) !important; }

    /* Result cards */
    .result-card {
        background-color: var(--card);
        border: 1px solid var(--border);
        border-left: 4px solid var(--gold);
        border-radius: 8px;
        padding: 1.2rem 1.4rem;
        margin-bottom: 1rem;
    }
    .result-header {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        margin-bottom: 0.6rem;
    }
    .result-num {
        background-color: var(--gold);
        color: var(--navy);
        border-radius: 50%;
        width: 26px;
        height: 26px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 0.8rem;
        flex-shrink: 0;
    }
    .topic-pill {
        background-color: var(--card2);
        color: var(--gold2);
        border: 1px solid var(--gold);
        border-radius: 20px;
        padding: 2px 10px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .result-title {
        font-weight: 700;
        color: var(--light);
        font-size: 1rem;
    }
    .relevance-badge {
        margin-left: auto;
        color: var(--gold);
        font-size: 0.85rem;
        font-weight: 600;
    }
    .result-text {
        color: #c8c0b0;
        font-size: 0.9rem;
        line-height: 1.6;
        margin-top: 0.5rem;
        border-top: 1px solid var(--border);
        padding-top: 0.6rem;
    }

    /* Hero card */
    .hero-card {
        background: linear-gradient(135deg, var(--card) 0%, var(--card2) 100%);
        border: 1px solid var(--border);
        border-top: 4px solid var(--gold);
        border-radius: 12px;
        padding: 2rem;
        margin-bottom: 2rem;
        text-align: center;
    }
    .hero-card h2 { margin-top: 0; }

    /* Feature cards */
    .feature-card {
        background-color: var(--card);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
        height: 100%;
    }
    .feature-icon {
        font-size: 2rem;
        margin-bottom: 0.5rem;
    }
    .feature-title {
        color: var(--gold);
        font-weight: 700;
        font-size: 1rem;
        margin-bottom: 0.3rem;
    }
    .feature-desc {
        color: var(--muted);
        font-size: 0.85rem;
    }

    /* Pipeline steps */
    .pipeline-step {
        background-color: var(--card);
        border: 1px solid var(--gold);
        border-radius: 8px;
        padding: 0.6rem 1rem;
        text-align: center;
        color: var(--light);
        font-size: 0.9rem;
        font-weight: 500;
    }
    .pipeline-arrow {
        text-align: center;
        color: var(--gold);
        font-size: 1.3rem;
        margin: 0.2rem 0;
    }

    /* Tech stack card */
    .tech-card {
        background-color: var(--card);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 0.7rem;
    }
    .tech-name {
        color: var(--gold);
        font-weight: 700;
        font-size: 0.95rem;
    }
    .tech-desc {
        color: var(--muted);
        font-size: 0.82rem;
        margin-top: 0.2rem;
    }
    </style>
    """, unsafe_allow_html=True)

# ─── Document Corpus ──────────────────────────────────────────────────────────

def get_documents():
    """Return the 13-document startup knowledge corpus."""
    return [
        Document(
            page_content=(
                "Product-market fit (PMF) is the degree to which a product satisfies a strong market demand. "
                "Coined by Marc Andreessen, PMF is often described as the moment when a startup's product "
                "genuinely resonates with its target customers — users come back repeatedly, tell others, "
                "and would be significantly disappointed if the product disappeared.\n\n"
                "Measuring PMF is more art than science. Sean Ellis proposed that a product has achieved "
                "PMF if at least 40% of surveyed users say they would be 'very disappointed' if they could "
                "no longer use it. Other signals include low churn, high Net Promoter Score (NPS), organic "
                "word-of-mouth growth, and customers who are actively using the product more over time.\n\n"
                "Founders who have not yet found PMF should focus entirely on discovering it before scaling. "
                "Pouring marketing spend into a product that lacks PMF only accelerates the burn rate without "
                "generating sustainable growth. The search for PMF typically involves deep customer interviews, "
                "rapid experimentation, and a willingness to pivot the core value proposition until users "
                "genuinely love what you're building."
            ),
            metadata={"topic": "product_market_fit", "title": "Product-Market Fit"}
        ),
        Document(
            page_content=(
                "A Minimum Viable Product (MVP) is the simplest version of a product that allows a team to "
                "collect the maximum amount of validated learning about customers with the least effort. "
                "The concept, popularized by Eric Ries in 'The Lean Startup', challenges the traditional "
                "notion of building a complete product before releasing it to the market.\n\n"
                "An MVP is not a buggy or incomplete product — it is a deliberately scoped product that "
                "tests a specific hypothesis about customer behavior or willingness to pay. Famous MVPs "
                "include Dropbox's explainer video (which drove 70,000 signups before the product was "
                "built), Airbnb's simple webpage with photos of the founders' apartment, and Zappos's "
                "manual order fulfillment before building a real e-commerce platform.\n\n"
                "The MVP mindset pushes teams to answer the question: what is the riskiest assumption we're "
                "making, and what is the smallest experiment we can run to test it? This prevents over-engineering "
                "and ensures that product decisions are grounded in real user feedback rather than internal "
                "assumptions. Iterating quickly from MVP to MVP is the core engine of lean product development."
            ),
            metadata={"topic": "mvp", "title": "Minimum Viable Product (MVP)"}
        ),
        Document(
            page_content=(
                "The Lean Startup methodology, developed by Eric Ries, applies lean manufacturing principles "
                "to the process of building startups. Its core engine is the Build-Measure-Learn feedback loop: "
                "build a small experiment, measure how customers respond, and use that data to decide whether "
                "to persevere with the current strategy or pivot to a new one.\n\n"
                "Central to lean startup thinking is the concept of validated learning — progress measured not "
                "in features shipped or lines of code written, but in hypotheses tested and customer insights "
                "gained. Vanity metrics like page views or total registered users often obscure whether a "
                "business is actually creating value for customers. Actionable metrics, such as activation rate "
                "or revenue per user, provide a more honest picture.\n\n"
                "Innovation accounting is the lean startup's answer to traditional financial accounting for "
                "early-stage companies. It establishes baseline metrics, tunes the engine by running experiments, "
                "and then decides whether to pivot or persevere based on whether the needle is moving in the "
                "right direction. This framework helps teams avoid building the wrong product efficiently — one "
                "of the most common and costly startup failure modes."
            ),
            metadata={"topic": "lean_startup", "title": "Lean Startup Methodology"}
        ),
        Document(
            page_content=(
                "Venture capital (VC) is a form of private equity financing provided by investors to startups "
                "and early-stage companies with high growth potential. VC firms raise funds from limited partners "
                "(LPs) — typically institutional investors like pension funds, university endowments, and family "
                "offices — and deploy that capital into a portfolio of startups in exchange for equity.\n\n"
                "VC investing follows a power law: the majority of returns come from a small number of "
                "breakout investments. This model means investors must accept that most portfolio companies will "
                "return little or nothing, while a few will generate 100x or more returns. This dynamic "
                "incentivizes VCs to fund bold, high-risk ideas that have the potential for massive scale — "
                "making VC unsuitable for lifestyle businesses or companies with moderate growth ceilings.\n\n"
                "The VC process typically moves through seed, Series A, B, and later rounds. Each round involves "
                "due diligence, term sheet negotiation, and the issuance of preferred shares with protective "
                "provisions. Founders should understand that raising VC is not free money — it is an exchange of "
                "equity and control for capital, and comes with the expectation of a large exit through an IPO "
                "or acquisition."
            ),
            metadata={"topic": "venture_capital", "title": "Venture Capital"}
        ),
        Document(
            page_content=(
                "Bootstrapping refers to building and growing a company using only personal savings, revenue "
                "generated by the business, and occasionally small loans — without taking on outside investment. "
                "Bootstrapped founders retain full equity and control of their company, and are accountable only "
                "to their customers rather than to investors.\n\n"
                "The bootstrapping model forces a healthy discipline: you must generate revenue early, keep costs "
                "low, and prioritize profitability over growth at all costs. Many highly successful companies "
                "were bootstrapped, including Mailchimp, Basecamp, and GitHub (before its eventual VC raise). "
                "The absence of investor pressure can also allow founders to build more deliberately and "
                "maintain a long-term vision without being pushed toward rapid scaling.\n\n"
                "The primary challenge of bootstrapping is pace. Without external capital, growth may be slower, "
                "making it harder to capture markets quickly in winner-take-most dynamics. Bootstrapped founders "
                "must also fund their own salaries, which creates personal financial pressure. The decision to "
                "bootstrap vs. raise is ultimately a question of the type of company you want to build and the "
                "market dynamics you're operating in."
            ),
            metadata={"topic": "bootstrapping", "title": "Bootstrapping"}
        ),
        Document(
            page_content=(
                "Burn rate is the rate at which a company spends its cash reserves before becoming profitable. "
                "Gross burn rate refers to total monthly operating expenses, while net burn rate accounts for "
                "any revenue coming in. For example, a company spending $200,000/month with $50,000 in revenue "
                "has a net burn rate of $150,000/month.\n\n"
                "Runway is the companion metric to burn rate: it measures how many months a company can continue "
                "operating before running out of money. Runway = Cash on Hand / Net Burn Rate. Investors and "
                "founders closely monitor runway, and it is generally advisable to start fundraising when you "
                "have 6–9 months of runway remaining — not when you're down to 2 months.\n\n"
                "Managing burn rate is a core survival skill for startup founders. Common strategies include "
                "delaying hires, negotiating deferred compensation, optimizing cloud infrastructure spend, and "
                "ruthlessly cutting features or programs that don't directly contribute to growth or revenue. "
                "In a down market, investors reward capital efficiency, and a low burn rate relative to progress "
                "made can be a significant competitive advantage during fundraising."
            ),
            metadata={"topic": "burn_rate", "title": "Burn Rate & Runway"}
        ),
        Document(
            page_content=(
                "A pivot is a structured course correction designed to test a new fundamental hypothesis about "
                "the product, business model, or growth engine. The term, popularized by Eric Ries, refers to "
                "a deliberate strategic change rather than random flailing. Successful pivots are grounded in "
                "learning from real customer data, not gut feelings.\n\n"
                "There are many types of pivots: a zoom-in pivot narrows the focus to a single feature that "
                "becomes the whole product; a customer segment pivot re-targets the product to a different "
                "audience; a platform pivot transitions from an application to a platform (or vice versa); "
                "and a business model pivot changes the revenue model entirely. YouTube began as a video dating "
                "site before pivoting to general video sharing. Slack was originally a gaming company.\n\n"
                "Knowing when to pivot vs. persevere is one of the hardest judgments a founder makes. Persevering "
                "too long on a broken strategy wastes runway; pivoting too early means abandoning a strategy "
                "before it has been fully tested. The key is to run experiments that generate clear signal — "
                "and to have the intellectual honesty to act on what the data says, even when it means "
                "abandoning months of work."
            ),
            metadata={"topic": "pivot", "title": "The Pivot"}
        ),
        Document(
            page_content=(
                "A go-to-market (GTM) strategy is the plan a company uses to bring a product to market and "
                "reach its target customers. It encompasses the target audience, value proposition, pricing "
                "model, distribution channels, and sales and marketing tactics. A strong GTM strategy is as "
                "important as the product itself — many technically superior products have failed due to "
                "poor market entry execution.\n\n"
                "GTM strategies vary widely by market type. Product-led growth (PLG) relies on the product "
                "itself as the primary driver of acquisition and expansion — Slack, Dropbox, and Figma are "
                "canonical examples. Sales-led growth relies on outbound and inbound sales teams to close "
                "enterprise deals. Community-led and content-led growth leverage organic communities and "
                "educational content to build trust and drive inbound demand.\n\n"
                "Key GTM components include the Ideal Customer Profile (ICP), which defines the specific "
                "company or individual most likely to buy; the positioning statement, which articulates why "
                "your product is different and better for that customer; and the sales motion, which defines "
                "the steps from first touch to closed deal. GTM strategy should be revisited at each growth "
                "stage as the customer profile and competitive landscape evolve."
            ),
            metadata={"topic": "go_to_market", "title": "Go-To-Market Strategy"}
        ),
        Document(
            page_content=(
                "Unit economics refers to the direct revenues and costs associated with a single unit of a "
                "business — usually one customer. The two most important unit economics metrics for startups "
                "are Customer Acquisition Cost (CAC) and Customer Lifetime Value (LTV). CAC is the total cost "
                "of acquiring a new customer, including all sales and marketing spend. LTV is the total net "
                "revenue a customer generates over their entire relationship with the company.\n\n"
                "Healthy unit economics typically require an LTV:CAC ratio of at least 3:1, meaning for every "
                "dollar spent acquiring a customer, the company expects to earn at least three dollars back. "
                "The CAC Payback Period — how many months it takes to recover the acquisition cost — is also "
                "watched closely by investors. SaaS businesses typically target a payback period under 12 months.\n\n"
                "Poor unit economics is one of the most common reasons startups fail or struggle to raise "
                "capital. If it costs more to acquire a customer than they're worth, no amount of scale will "
                "fix the underlying problem. Improving unit economics usually involves reducing CAC through "
                "more efficient marketing channels, increasing LTV through better retention or upsell, or "
                "both. Understanding unit economics forces founders to think rigorously about the economics "
                "of their growth."
            ),
            metadata={"topic": "unit_economics", "title": "Unit Economics"}
        ),
        Document(
            page_content=(
                "Growth hacking is a marketing approach focused on rapidly identifying the most effective and "
                "efficient ways to grow a business. The term was coined by Sean Ellis in 2010 to describe "
                "a mindset where every decision is oriented toward growth, blending product development, "
                "data analysis, and marketing into a single function.\n\n"
                "Classic growth hacks include Hotmail's 'PS: Get your free email at Hotmail' email signature, "
                "Dropbox's referral program (which drove a 3,900% increase in signups), and Airbnb's "
                "Craigslist integration that allowed hosts to cross-post listings to a massive existing "
                "audience. What these examples share is creative exploitation of existing platforms, networks, "
                "or user behaviors to drive viral or near-viral growth.\n\n"
                "Modern growth hacking is deeply data-driven. Growth teams run continuous A/B experiments "
                "across the funnel — from ad creative and landing page copy to onboarding flows and email "
                "sequences. The AARRR framework (Acquisition, Activation, Retention, Referral, Revenue) "
                "provides a structure for identifying which part of the funnel to focus on. Growth hacking "
                "is most powerful after product-market fit has been found; applying it too early risks "
                "scaling a broken product."
            ),
            metadata={"topic": "growth_hacking", "title": "Growth Hacking"}
        ),
        Document(
            page_content=(
                "Founder equity refers to the ownership stake held by the founders of a company. At inception, "
                "founders typically own 100% of the business, but this stake is diluted through subsequent "
                "funding rounds, employee option pools, and advisor grants. Managing dilution thoughtfully "
                "is critical to ensuring founders remain sufficiently motivated and rewarded for building "
                "long-term value.\n\n"
                "Vesting schedules are the standard mechanism for protecting both the company and co-founders. "
                "A typical arrangement is a 4-year vest with a 1-year cliff: no shares vest in the first year, "
                "then 25% vest on the cliff date, and the remaining shares vest monthly over the following "
                "36 months. This ensures that founders must remain committed to the business to earn their "
                "full equity stake, and protects co-founders and investors if one founder departs early.\n\n"
                "Founder dilution becomes a significant concern as companies raise multiple rounds. An early "
                "seed investor taking 20% may seem modest, but by Series B, the cumulative effect of option "
                "pools and subsequent rounds can leave founders with surprisingly small stakes. Using cap "
                "table simulation tools before signing term sheets helps founders understand the long-term "
                "implications of their financing decisions."
            ),
            metadata={"topic": "founder_equity", "title": "Founder Equity & Vesting"}
        ),
        Document(
            page_content=(
                "A term sheet is a non-binding document that outlines the key terms and conditions under which "
                "an investor will make an investment in a startup. It serves as the basis for drafting the "
                "definitive legal agreements — the Stock Purchase Agreement, Investors' Rights Agreement, and "
                "related documents. While non-binding, term sheets set the tone and framework for the entire "
                "investment relationship.\n\n"
                "Key economic terms include the pre-money valuation (the company's value before the investment), "
                "the investment amount, and the resulting ownership percentage. Liquidation preferences determine "
                "how proceeds are distributed in an exit: a 1x non-participating liquidation preference means "
                "investors get their money back before common shareholders receive anything, but don't "
                "participate further. Participating preferred shares allow investors to recoup their investment "
                "AND share in remaining proceeds, which can dramatically reduce founder payouts in modest exits.\n\n"
                "Control terms are equally important and often under-scrutinized by first-time founders. "
                "Protective provisions give investors veto rights over major decisions — raising new capital, "
                "selling the company, or changing the capital structure. Pro-rata rights allow investors to "
                "maintain their ownership percentage in future rounds. Anti-dilution provisions protect "
                "investors if the company raises money at a lower valuation in the future. Founders should "
                "work with experienced startup lawyers to fully understand the implications of every term."
            ),
            metadata={"topic": "term_sheets", "title": "Term Sheets"}
        ),
        Document(
            page_content=(
                "Customer discovery is the first phase of the customer development process, developed by "
                "Steve Blank. Its goal is to get out of the building and test whether the problem you "
                "hypothesize exists is real, and whether the solution you're proposing actually addresses "
                "it. The phase ends when you've validated your core business model hypotheses with real "
                "customer data.\n\n"
                "Effective customer discovery interviews are non-leading conversations focused on understanding "
                "the customer's world, not pitching your solution. The Mom Test, a framework popularized by "
                "Rob Fitzpatrick, offers a simple rule: ask about real past behavior, not hypothetical future "
                "behavior. 'Would you pay for this?' is a bad question; 'Tell me about the last time you "
                "tried to solve this problem and what happened' is a good one.\n\n"
                "The output of customer discovery is a refined set of hypotheses about customer segments, "
                "pain points, and willingness to pay — not a list of feature requests. Customers are "
                "notoriously poor at specifying what they want; they are excellent at describing their "
                "problems and frustrations. The founder's job is to translate those insights into product "
                "decisions. Most successful startups report conducting 50–100 customer discovery interviews "
                "before writing a single line of production code."
            ),
            metadata={"topic": "customer_discovery", "title": "Customer Discovery"}
        ),
    ]

# ─── Embeddings ───────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading embedding model...")
def get_embeddings():
    """Load the sentence-transformer embedding model (cached for session lifetime)."""
    return HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

# ─── Vector Store ─────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Initialising knowledge base...")
def get_vector_store(_embeddings):
    """Load or build the ChromaDB vector store. Persists to disk automatically."""
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    existing = [c.name for c in chroma_client.list_collections()]

    if COLLECTION_NAME in existing:
        # Collection already on disk — load without re-embedding
        vectorstore = Chroma(
            client=chroma_client,
            collection_name=COLLECTION_NAME,
            embedding_function=_embeddings,
        )
    else:
        # First run — chunk, embed, and persist
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )
        raw_docs = get_documents()
        chunks = splitter.split_documents(raw_docs)
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=_embeddings,
            client=chroma_client,
            collection_name=COLLECTION_NAME,
        )

    return vectorstore

# ─── Pages ────────────────────────────────────────────────────────────────────

def page_home():
    st.title("Startup Knowledge Base")

    st.markdown("""
    <div class="hero-card">
        <div style="font-size:2.5rem; margin-bottom:0.5rem;">📚</div>
        <h2>Semantic Search for Startup Concepts</h2>
        <p style="color:#8a9ab5; margin:0;">
            Ask any question about startups, venture capital, product strategy, or growth —
            and instantly surface the most relevant knowledge from our curated corpus.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    features = [
        ("🔍", "Search", "Ask natural language questions and get semantically relevant answers from 13 expert documents."),
        ("📖", "About", "Learn what Retrieval-Augmented Generation (RAG) is and how this app works under the hood."),
        ("📊", "Statistics", "Explore the corpus — see document count, chunk breakdown, and all topics covered."),
    ]
    for col, (icon, title, desc) in zip([col1, col2, col3], features):
        with col:
            st.markdown(f"""
            <div class="feature-card">
                <div class="feature-icon">{icon}</div>
                <div class="feature-title">{title}</div>
                <div class="feature-desc">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="background-color:#16263b; border:1px solid #2a4a6b; border-radius:8px; padding:1rem 1.4rem;">
        <strong style="color:#c9a84c;">Quick Start:</strong>
        <span style="color:#c8c0b0;">
        Navigate to <strong>Search</strong> in the sidebar and type a question like
        <em>"How do I find product-market fit?"</em> or <em>"What is a good LTV to CAC ratio?"</em>
        </span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### Topics Covered")
    topics = [doc.metadata["title"] for doc in get_documents()]
    cols = st.columns(3)
    for i, topic in enumerate(topics):
        with cols[i % 3]:
            st.markdown(f"<span class='topic-pill'>✦ {topic}</span> &nbsp;", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)


def page_search():
    st.title("Search Startup Knowledge")
    st.markdown("<p style='color:#8a9ab5; margin-top:-0.5rem;'>Ask any startup or business question in plain English.</p>", unsafe_allow_html=True)

    query = st.text_input(
        "Your question",
        placeholder="e.g. How do I know when I've found product-market fit?",
        label_visibility="collapsed",
    )

    col_a, col_b = st.columns([3, 1])
    with col_b:
        k = st.slider("Results", min_value=1, max_value=10, value=5)
    with col_a:
        search_btn = st.button("Search", use_container_width=True)

    if (search_btn or query) and query.strip():
        embeddings = get_embeddings()
        vectorstore = get_vector_store(embeddings)

        with st.spinner("Searching knowledge base..."):
            results = vectorstore.similarity_search_with_score(query.strip(), k=k)

        if not results:
            st.warning("No results found. Try a different query.")
            return

        st.markdown(f"<h3 style='margin-top:1.5rem;'>Top {len(results)} Results</h3>", unsafe_allow_html=True)

        for i, (doc, score) in enumerate(results):
            # Convert L2 distance to a 0–100% relevance score
            # With normalized embeddings, L2 distance ranges ~0–2; map to 100%–0%
            relevance = max(0.0, round((1 - score / 2) * 100, 1))
            topic_label = doc.metadata.get("topic", "unknown").replace("_", " ").title()
            title = doc.metadata.get("title", topic_label)
            text = doc.page_content.replace("\n", " ")

            st.markdown(f"""
            <div class="result-card">
                <div class="result-header">
                    <span class="result-num">{i+1}</span>
                    <span class="topic-pill">{topic_label}</span>
                    <span class="result-title">{title}</span>
                    <span class="relevance-badge">⬥ {relevance}% match</span>
                </div>
                <div class="result-text">{text}</div>
            </div>
            """, unsafe_allow_html=True)
    elif search_btn and not query.strip():
        st.warning("Please enter a search query.")


def page_about():
    st.title("About This App")
    st.markdown("""
    <p style='color:#8a9ab5;'>How Retrieval-Augmented Generation (RAG) works — and what's under the hood.</p>
    """, unsafe_allow_html=True)

    st.markdown("## What is RAG?")
    st.markdown("""
    Retrieval-Augmented Generation (RAG) is an AI architecture pattern that improves the quality and
    accuracy of language model responses by grounding them in a retrieved set of relevant documents.
    Rather than relying solely on knowledge baked into a model's parameters during training, a RAG
    system first searches a knowledge base for the most relevant passages, then uses those passages
    to inform the final response.

    This app demonstrates the **retrieval** half of RAG: given a natural language query, it uses
    semantic vector search to surface the most relevant chunks from a curated startup knowledge corpus.
    The embeddings capture meaning rather than keywords, so a query like *"how do I avoid running out
    of money"* will correctly surface documents about burn rate and runway — even without those exact words.
    """)

    st.markdown("## How This App Works")

    pipeline = [
        ("1", "Your Query", "You type a natural language question."),
        ("2", "Embedding Model", "all-MiniLM-L6-v2 converts your query into a 384-dimensional vector."),
        ("3", "Vector Search", "ChromaDB finds the k nearest document chunks by cosine similarity."),
        ("4", "Result Ranking", "Chunks are ranked by relevance score and returned with metadata."),
        ("5", "Display", "Results are shown with topic labels, titles, and relevance percentages."),
    ]

    for step in pipeline:
        num, title, desc = step
        st.markdown(f"""
        <div style="display:flex; align-items:flex-start; gap:1rem; margin-bottom:0.7rem;">
            <div style="background:#c9a84c; color:#0d1b2a; border-radius:50%; width:28px; height:28px;
                        display:flex; align-items:center; justify-content:center; font-weight:700;
                        font-size:0.8rem; flex-shrink:0; margin-top:3px;">{num}</div>
            <div>
                <div style="color:#e8c97a; font-weight:700;">{title}</div>
                <div style="color:#c8c0b0; font-size:0.88rem;">{desc}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("## Tech Stack")

    col1, col2 = st.columns(2)
    stack = [
        ("Streamlit", "Web UI framework for Python — builds interactive data apps with minimal code."),
        ("LangChain", "Orchestration framework for LLM and retrieval pipelines; handles document splitting."),
        ("ChromaDB", "Open-source vector database with on-disk persistence and fast nearest-neighbor search."),
        ("sentence-transformers", "Lightweight, fast transformer models optimised for semantic similarity tasks."),
        ("all-MiniLM-L6-v2", "384-dim embedding model: 6-layer MiniLM fine-tuned on 1B+ sentence pairs."),
        ("Python 3.11", "Runtime environment; deployed on Render via render.yaml configuration."),
    ]
    for i, (name, desc) in enumerate(stack):
        with (col1 if i % 2 == 0 else col2):
            st.markdown(f"""
            <div class="tech-card">
                <div class="tech-name">⬥ {name}</div>
                <div class="tech-desc">{desc}</div>
            </div>
            """, unsafe_allow_html=True)


def page_statistics():
    st.title("Corpus Statistics")
    st.markdown("<p style='color:#8a9ab5; margin-top:-0.5rem;'>An overview of the knowledge base.</p>", unsafe_allow_html=True)

    embeddings = get_embeddings()
    vectorstore = get_vector_store(embeddings)
    total_chunks = vectorstore._collection.count()
    raw_docs = get_documents()

    # Metric cards
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Documents", len(raw_docs))
    with col2:
        st.metric("Total Chunks", total_chunks)
    with col3:
        st.metric("Chunk Size", f"{CHUNK_SIZE} / {CHUNK_OVERLAP}")

    st.markdown("<br>", unsafe_allow_html=True)

    # Topics table
    st.markdown("### Topics Covered")
    rows = []
    for doc in raw_docs:
        rows.append({
            "Title": doc.metadata["title"],
            "Topic Slug": doc.metadata["topic"],
            "Characters": len(doc.page_content),
            "Preview": doc.page_content[:120].replace("\n", " ") + "...",
        })
    df = pd.DataFrame(rows)
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Title": st.column_config.TextColumn("Title", width="medium"),
            "Topic Slug": st.column_config.TextColumn("Topic Slug", width="medium"),
            "Characters": st.column_config.NumberColumn("Characters", width="small"),
            "Preview": st.column_config.TextColumn("Content Preview", width="large"),
        }
    )

    # Bar chart
    st.markdown("### Document Length (characters)")
    chart_df = pd.DataFrame({
        "Topic": [d.metadata["title"] for d in raw_docs],
        "Characters": [len(d.page_content) for d in raw_docs],
    }).set_index("Topic")
    st.bar_chart(chart_df, color="#c9a84c")

    # Config info
    st.markdown("### Chunking Configuration")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric("Chunk Size", CHUNK_SIZE)
    with col_b:
        st.metric("Chunk Overlap", CHUNK_OVERLAP)
    with col_c:
        st.metric("Embedding Model", "MiniLM-L6-v2")

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    # st.set_page_config MUST be the first Streamlit call
    st.set_page_config(
        page_title="Startup Knowledge Base",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_css()

    pages = {
        "🏠 Home":       page_home,
        "🔍 Search":     page_search,
        "📖 About":      page_about,
        "📊 Statistics": page_statistics,
    }

    with st.sidebar:
        st.markdown("""
        <div style="padding: 1rem 0 1.5rem 0; border-bottom: 1px solid #2a4a6b; margin-bottom: 1rem;">
            <div style="font-size:1.4rem; font-weight:700; color:#c9a84c;">📚 Startup KB</div>
            <div style="font-size:0.75rem; color:#8a9ab5; margin-top:2px;">Knowledge Base · RAG Demo</div>
        </div>
        """, unsafe_allow_html=True)
        selection = st.radio(
            "Navigation",
            list(pages.keys()),
            label_visibility="collapsed",
        )

    pages[selection]()


if __name__ == "__main__":
    main()
