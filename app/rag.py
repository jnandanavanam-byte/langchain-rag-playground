"""
RAG Comparison Engine

Manages:

Pipeline 1
vs
Pipeline 2

Future Enhancements:
- RunnableParallel
- Streaming Responses
- Multi-Pipeline Evaluation
"""


from app.pipeline import (
    PipelineConfig,
    RagPipeline
)


# ============================================================
# RAG Comparison Engine
# ============================================================

class RagComparison:

    def __init__(
        self,
        pipeline1_config: PipelineConfig,
        pipeline2_config: PipelineConfig
    ):
        """
        Initialize two RAG pipelines for comparison.
        """

        self.pipeline1 = RagPipeline(
            pipeline1_config
        )

        self.pipeline2 = RagPipeline(
            pipeline2_config
        )

    # ========================================================
    # Build Pipelines
    # ========================================================

    def build(
        self,
        documents
    ):
        """
        Build both pipelines.
        """

        pipeline1_stats = (
            self.pipeline1.build(
                documents
            )
        )

        pipeline2_stats = (
            self.pipeline2.build(
                documents
            )
        )

        return {
            "pipeline1": pipeline1_stats,
            "pipeline2": pipeline2_stats
        }

    # ========================================================
    # Ask Question
    # ========================================================

    def ask(
        self,
        question
    ):
        """
        Execute query against both pipelines.
        """

        response1 = (
            self.pipeline1.ask(
                question
            )
        )

        response2 = (
            self.pipeline2.ask(
                question
            )
        )

        return {
            "pipeline1": response1,
            "pipeline2": response2
        }

    # ========================================================
    # Status
    # ========================================================

    def is_ready(self):
        """
        Check whether both pipelines are ready.
        """

        return (
            self.pipeline1.is_ready()
            and
            self.pipeline2.is_ready()
        )

    # ========================================================
    # Configuration
    # ========================================================

    def get_configuration(self):
        """
        Return pipeline configurations.
        """

        return {
            "pipeline1":
                self.pipeline1.get_configuration(),

            "pipeline2":
                self.pipeline2.get_configuration()
        }

    # ========================================================
    # Statistics
    # ========================================================

    def get_statistics(self):
        """
        Return pipeline statistics.
        """

        return {
            "pipeline1":
                self.pipeline1.get_statistics(),

            "pipeline2":
                self.pipeline2.get_statistics()
        }

    # ========================================================
    # String Representation
    # ========================================================

    def __repr__(self):

        return (
            "RagComparison("
            "pipeline1="
            f"{self.pipeline1.get_configuration()}, "
            "pipeline2="
            f"{self.pipeline2.get_configuration()}"
            ")"
        )