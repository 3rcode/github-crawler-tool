import os
from preprocessing import score_by_bleu


if __name__ == '__main__':
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    changelog_path = os.path.join(ROOT_DIR, 'changelog.txt')
    with open(changelog_path, 'r', encoding='utf-8') as f:
        changelog = f.read()

    commit1 = "feat: session.resolveHost"
    commit2 = "fix: showAboutPanel also on linux (#37828) showAboutPanel also on linux"
    score1_a = score_by_bleu(commit1, changelog)
    score2_a = score_by_bleu(commit2, changelog)
    print(score1_a, score2_a)
    # OUTPUT:
    '''('session resolveproxy 17222', 0.5671687320916967) 
       ('fix bug where app would crash if app showaboutpanel 
         was call before set any about panel options on linux 19625', 0.35492629278049054)
    '''

    '''Note: Need to restructure commit.txt file to get better results.'''