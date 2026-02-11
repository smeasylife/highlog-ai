"""LangGraph 시각화 스크립트

면접 질문 생성 그래프와 실시간 면접 그래프를 시각화합니다.
"""
import asyncio
from pathlib import Path
from app.graphs.interview_graph import interview_graph
from app.graphs.record_analysis import question_generation_graph


def visualize_interview_graph():
    """면접 그래프 시각화"""
    print("=" * 60)
    print("📊 인터뷰 그래프 시각화 중...")
    print("=" * 60)

    # 1. ASCII 아트 (가장 확실한 방법)
    try:
        print("🎨 ASCII 아트 생성 중...")
        ascii_art = interview_graph.graph.get_graph().draw_ascii()
        with open("docs/interview_graph_ascii.txt", "w", encoding="utf-8") as f:
            f.write(ascii_art)
        print("✅ interview_graph_ascii.txt 저장 완료")

        # 콘솔에도 출력
        print("\n" + "=" * 60)
        print("📊 인터뷰 그래프 (ASCII)")
        print("=" * 60)
        print(ascii_art)
    except Exception as e:
        print(f"❌ ASCII 아트 생성 실패: {e}")

    # 2. Mermaid PNG
    try:
        print("\n📸 Mermaid PNG 생성 중...")
        interview_graph.graph.get_graph().draw_mermaid_png(
            output_file_path=Path("docs/interview_graph.png")
        )
        print("✅ interview_graph.png 저장 완료")
    except Exception as e:
        print(f"❌ PNG 생성 실패: {e}")
        print("   (참고: PNG 생성은 graphviz가 필요할 수 있습니다)")

    print()


def visualize_question_generation_graph():
    """질문 생성 그래프 시각화"""
    print("=" * 60)
    print("📊 질문 생성 그래프 시각화 중...")
    print("=" * 60)

    # 1. ASCII 아트
    try:
        print("🎨 ASCII 아트 생성 중...")
        ascii_art = question_generation_graph.graph.get_graph().draw_ascii()
        with open("docs/question_generation_graph_ascii.txt", "w", encoding="utf-8") as f:
            f.write(ascii_art)
        print("✅ question_generation_graph_ascii.txt 저장 완료")

        # 콘솔에도 출력
        print("\n" + "=" * 60)
        print("📊 질문 생성 그래프 (ASCII)")
        print("=" * 60)
        print(ascii_art)
    except Exception as e:
        print(f"❌ ASCII 아트 생성 실패: {e}")

    # 2. Mermaid PNG
    try:
        print("\n📸 Mermaid PNG 생성 중...")
        question_generation_graph.graph.get_graph().draw_mermaid_png(
            output_file_path=Path("docs/question_generation_graph.png")
        )
        print("✅ question_generation_graph.png 저장 완료")
    except Exception as e:
        print(f"❌ PNG 생성 실패: {e}")
        print("   (참고: PNG 생성은 graphviz가 필요할 수 있습니다)")

    print()


def print_graph_info():
    """그래프 정보 출력"""
    print("=" * 60)
    print("📋 그래프 구조 정보")
    print("=" * 60)

    # 인터뷰 그래프
    print("\n🎤 인터뷰 그래프:")
    try:
        drawable = interview_graph.graph.get_graph()
        print(f"  노드 수: {len(list(drawable.nodes))}")
        print(f"  노드: {list(drawable.nodes)}")
    except Exception as e:
        print(f"  에러: {e}")

    # 질문 생성 그래프
    print("\n📝 질문 생성 그래프:")
    try:
        drawable = question_generation_graph.graph.get_graph()
        print(f"  노드 수: {len(list(drawable.nodes))}")
        print(f"  노드: {list(drawable.nodes)}")
    except Exception as e:
        print(f"  에러: {e}")

    print()


if __name__ == "__main__":
    # docs 폴더가 없으면 생성
    Path("docs").mkdir(exist_ok=True)

    # 그래프 정보 출력
    print_graph_info()

    # 시각화 실행
    visualize_question_generation_graph()
    visualize_interview_graph()

    print("=" * 60)
    print("✅ 모든 시각화가 완료되었습니다!")
    print("  - docs/ 폴더를 확인해주세요.")
    print("=" * 60)
