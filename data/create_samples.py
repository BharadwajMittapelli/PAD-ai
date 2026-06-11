import pandas as pd
import random

data = []

categories = {
    "Authority/Urgency manipulation": [
        ("https://admin.o365-verify-update.com/auth", "Subject: URGENT: Final Notice Before Account Deletion\n\nYour account has been flagged for violation of our terms. You have 24 hours to review the violations and appeal. Click the link to review.", "Threat of account deletion combined with a plausible-looking admin domain."),
        ("https://paypal-resolution-center.net/case/88123", "Subject: Resolution Center: Unauthorized Login Attempt\n\nWe detected a login from a new device in Russia. If this wasn't you, please secure your account immediately by logging in.", "Fear-inducing geolocation alert with a fake resolution center URL."),
        ("https://irs-tax-refund-portal.gov.us-auth.com", "Subject: IRS Notice: Your Tax Refund is Pending\n\nYour recent tax return has been processed. However, we need additional verification to release your funds. Please complete the form within 48 hours.", "Authority impersonation (IRS) using a complex subdomain structure to hide the true domain."),
        ("https://aws-billing-update.amazon-aws-secure.com", "Subject: Action Required: AWS Account Suspension Notice\n\nYour AWS payment method has expired. All running EC2 instances will be terminated in 12 hours unless you update your billing details.", "Targeting developers/admins with high-stakes infrastructure threats."),
        ("https://appleid.apple.com-recovery-session.xyz", "Subject: Your Apple ID was used to sign in to iCloud via a web browser\n\nYour Apple ID was used to sign in to iCloud on a Windows PC. If you did not sign in, you should change your password immediately.", "Exact copy of legitimate Apple alert but with a deceiving subdomain."),
        ("https://secure.chase-banking-alert.com/login", "Subject: Security Alert: Temporary Hold on Your Chase Account\n\nWe have placed a temporary hold on your account due to suspicious transactions. Please verify your recent activity to lift the hold.", "Banking impersonation creating urgency to clear a hold."),
        ("https://workspace.google.com-admin-review.net", "Subject: Google Workspace: Domain Suspension Alert\n\nYour Google Workspace domain registration is missing payment. Services will be disrupted tomorrow. Update payment info now.", "Targeting business owners with domain suspension threats."),
        ("https://netflix-membership-renew.com", "Subject: Payment Declined: Your Netflix Membership is on hold\n\nWe couldn't process your last payment. Update your payment method to continue enjoying your favorite shows.", "Common consumer service urgency hook."),
        ("https://hr-portal-docusign.com/document/view", "Subject: MANDATORY: Updated Employee Code of Conduct (Signature Required)\n\nAll employees must sign the updated 2025 Code of Conduct by end of day. Failure to do so will result in HR action.", "Internal HR impersonation creating employment-related urgency."),
        ("https://zoom-meeting-invite.com/j/88312", "Subject: Missed Zoom Meeting: Urgent Project Update\n\nYou missed a critical project sync. The recording and action items are available here. Please review immediately.", "Exploiting FOMO and professional urgency regarding missed meetings.")
    ],
    "Subtle homoglyphs & obfuscated URLs": [
        ("https://www.rnicrosoft.com/login", "Subject: Security Update for Windows Defender\n\nPlease download the latest security patch for Windows Defender to stay protected against new threats.", "Uses 'rn' instead of 'm' (rnicrosoft)."),
        ("https://www.paypaI.com/signin", "Subject: Your account has been limited\n\nPlease verify your identity to restore full access to your PayPal account.", "Uses uppercase 'i' instead of lowercase 'L' (paypaI)."),
        ("https://www.g00gle.com/accounts/recovery", "Subject: Critical security alert\n\nSomeone knows your password. Take action to secure your account.", "Uses zeroes instead of 'o' (g00gle)."),
        ("https://www.linĸedin.com/messages", "Subject: You appeared in 14 searches this week\n\nSee who's looking at your profile. You have a new message from a recruiter.", "Uses Cyrillic 'ĸ' instead of 'k'."),
        ("https://www.faceb00k.com/login", "Subject: Your page will be unpublished\n\nYour Facebook Page has been reported for community standards violations. Appeal now.", "Uses zeroes instead of 'o'."),
        ("https://www.citibañk.com/banking", "Subject: Statement Available\n\nYour monthly statement is ready to view. Log in to your account.", "Uses 'ñ' instead of 'n'."),
        ("https://www.dropb0x.com/s/file", "Subject: A document was shared with you\n\nYour colleague shared a confidential document via Dropbox.", "Uses zero instead of 'o'."),
        ("https://www.slack-w0rkspace.com/login", "Subject: Slack Workspace Deactivation Warning\n\nYour Slack workspace is inactive and will be archived. Log in to keep it active.", "Uses zero instead of 'o'."),
        ("https://www.github-rep0.com/auth", "Subject: Unauthorized access to your repository\n\nWe detected an unauthorized SSH key added to your GitHub account. Review it now.", "Uses zero instead of 'o'."),
        ("https://www.twiiter.com/login", "Subject: Unusual login attempt\n\nWe noticed a login from a new device. Is this you?", "Misspelling 'twiiter' instead of 'twitter'.")
    ],
    "Mixed legitimate + malicious content": [
        ("https://malicious-redirect.com?url=https://www.microsoft.com", "Subject: Welcome to your new Microsoft Services\n\nThanks for subscribing! To manage your subscription, please log in. You can also visit our official support page at https://support.microsoft.com.", "Mixes legitimate support links with a malicious redirect in the primary CTA."),
        ("https://fake-login-page.com/auth", "Subject: Invoice #8819 from Amazon.com\n\nYour order has shipped! Tracking number: 1z99999. If you need to return this item, visit our return center [Link]. For privacy policy, visit amazon.com/privacy.", "Includes actual Amazon privacy and terms links but a fake return center link."),
        ("https://secure-update-portal.com", "Subject: Important Update regarding your Chase Credit Card\n\nDear Customer, please update your profile. Chase Bank, N.A. Member FDIC. Equal Housing Lender. © 2025 JPMorgan Chase & Co.", "Includes legitimate-looking boilerplate, footers, and copyright text to bypass spam filters."),
        ("https://compromised-site.com/hidden-phish", "Subject: Weekly Newsletter - Tech Trends\n\nRead our latest article on AI advancements! Also, don't forget to update your subscriber preferences [Link].", "A legitimate newsletter template hijacked to include a single malicious link."),
        ("https://bit.ly/malicious-link", "Subject: Salesforce CRM Update Required\n\nSalesforce requires all users to update their passwords. Read the Salesforce Trust status at trust.salesforce.com. Update your password here: [Link]", "Combines real Salesforce trust URLs with a malicious shortlink."),
        ("https://fake-docusign-portal.com", "Subject: Completed: Please DocuSign: NDA_2025.pdf\n\nYour document has been completed. Powered by DocuSign. Visit docusign.com to learn more about electronic signatures.", "Mimics a completed DocuSign email with real DocuSign informational links."),
        ("https://malicious-app-auth.com", "Subject: Connect Zoom to your Google Calendar\n\nEnhance your productivity by connecting Zoom. Google Privacy Policy | Zoom Terms of Service. Connect here: [Link]", "Uses real privacy policy links from both vendors to add legitimacy."),
        ("https://phishing-site.com/verify", "Subject: Your LinkedIn Premium Subscription\n\nYour receipt is attached. If you wish to cancel, click here. LinkedIn Ireland Unlimited Company, Wilton Place, Dublin 2.", "Includes the real physical address and company name of the targeted brand."),
        ("https://fake-hr-survey.com", "Subject: 2025 Employee Engagement Survey\n\nPlease take a moment to complete our annual survey. Your feedback is anonymous. View our internal HR policies on the intranet.", "Mentions internal resources to sound like a legitimate internal communication."),
        ("https://malicious-update-server.com", "Subject: MacOS Sequoia 15.2 Update Available\n\nApple has released a critical security update. Learn more about Apple security updates at support.apple.com. Download the installer here.", "Mixes real Apple support knowledge base links with a fake download link.")
    ],
    "Low-entropy / conversational phishing": [
        ("https://tinyurl.com/meeting-notes-123", "Subject: Re: yesterday's meeting\n\nHey, can you review these notes when you have a sec? Let me know if I missed anything.", "Extremely short, conversational tone resembling a colleague. No urgent language."),
        ("https://bit.ly/invoice-q3", "Subject: invoice attached\n\nHi, please find the invoice for last month. Let me know when it's paid. Thanks.", "Casual lowercase subject and body, typical of quick internal emails."),
        ("https://goo.gl/shared-doc", "Subject: the document\n\nHere is the doc you asked for.", "Minimalist approach. Evades filters looking for complex phishing phrases."),
        ("https://t.co/presentation-draft", "Subject: slides\n\nDraft is ready, take a look before the call.", "Conversational and brief, assumes existing context."),
        ("https://is.gd/project-update", "Subject: Quick question\n\nAre we still on for later? Also, I updated the tracker here.", "Blends a scheduling question with a malicious link."),
        ("https://ow.ly/team-lunch", "Subject: lunch menu\n\nPlease pick what you want for the team lunch on Friday from this menu.", "Uses a mundane topic (team lunch) to lower defenses."),
        ("https://cutt.ly/new-policy", "Subject: fyi\n\nJust saw this, thought you should read it.", "Vague curiosity gap, very common in quick colleague interactions."),
        ("https://rebrand.ly/contact-info", "Subject: new phone number\n\nHey, I lost my phone. Can you save my new number? Details here.", "Personal conversational style, bypassing corporate filters."),
        ("https://short.io/vacation-pics", "Subject: pics\n\nFinally uploaded the photos from the trip!", "Exploits personal relationships and curiosity."),
        ("https://tiny.cc/article-link", "Subject: interesting read\n\nThought of you when I saw this article.", "Friendly sharing tone, no typical phishing keywords.")
    ],
    "Spear-phishing style (personalized)": [
        ("https://vendor-portal.com/invoice/ACME-Corp", "Subject: Outstanding Invoice for ACME Corp Services\n\nHi John, checking in on Invoice #4492 for the Q3 consulting services provided to ACME Corp. Please review the attached breakdown.", "Highly personalized, using fake company names and names (simulated)."),
        ("https://benefits-enrollment.com/employee/jdoe", "Subject: Action Required: Your 2025 Health Benefits Enrollment\n\nJane, your current health plan selection expires next week. Please confirm your dependents on the portal.", "Targets an individual's HR benefits with their name."),
        ("https://github-auth-update.com/user/devteam", "Subject: Security Notice for your GitHub Repo 'backend-api'\n\nAlex, we noticed a vulnerability in the dependencies of your 'backend-api' repository. Please apply the patch.", "Targets a developer using specific project names."),
        ("https://conference-registration.com/speaker/speaker-id", "Subject: Speaker Details for CyberSec Con 2025\n\nDear Dr. Smith, please confirm your speaking slot and upload your presentation slides to the speaker portal.", "Targets a professional attending a specific conference."),
        ("https://real-estate-docs.com/closing/123-Main-St", "Subject: Closing Documents for 123 Main St\n\nHi Sarah, the final closing documents for the property are ready for your review before tomorrow's signing.", "Targets someone in the middle of a real estate transaction."),
        ("https://school-portal-update.com/parent/student", "Subject: Urgent: Incident Report regarding your child\n\nDear Parent, please review the attached incident report from the principal's office regarding today's events.", "Targets a parent with alarming, personalized news."),
        ("https://tax-advisor-portal.com/client/2024-returns", "Subject: Missing W-2 for your 2024 Tax Return\n\nHi Michael, I'm reviewing your file and it looks like we're still missing your W-2 from your previous employer. Upload it here.", "Impersonates a tax advisor during tax season."),
        ("https://alumni-network.com/update-profile", "Subject: Update your alumni contact info - Class of 2015\n\nHi David, we are updating the alumni directory for the Class of 2015. Please ensure your details are correct.", "Uses educational background for personalization."),
        ("https://local-gym-membership.com/billing", "Subject: Issue with your monthly gym membership payment\n\nHi Emily, your card on file for the Downtown Fitness Center declined this morning. Please update it.", "Targets local service memberships."),
        ("https://freelance-platform.com/project/award", "Subject: You have been awarded the 'Website Redesign' project\n\nCongratulations Chris! The client has awarded you the project. Please accept the terms and begin work.", "Targets freelancers with fake project awards.")
    ],
    "Attachment/social engineering references": [
        ("https://secure-file-transfer.com/download/Invoice_PDF.exe", "Subject: FW: Unpaid Invoice - Second Notice\n\nPlease find the attached invoice. I cannot open the PDF, can you check if it works for you?", "Social engineering referencing an attachment that is actually a link to malware."),
        ("https://cloud-drive-share.com/view/document", "Subject: Scan from Office Printer (Scanner_001)\n\nYou have received a new scanned document from the office printer. Click to view or download.", "Impersonates automated office equipment notifications."),
        ("https://hr-secure-docs.com/view/salary-changes", "Subject: CONFIDENTIAL: 2025 Salary Adjustments\n\nAttached is the encrypted spreadsheet containing the proposed salary adjustments for your department.", "Uses curiosity and confidentiality to bait the user."),
        ("https://legal-document-review.com/subpoena", "Subject: LEGAL NOTICE: Subpoena to Testify\n\nYou have been served a subpoena. Please download the attached legal documents for your court date.", "Uses fear and legal authority."),
        ("https://courier-tracking.com/package/1Z999", "Subject: Delivery Attempt Failed - Package #1Z999\n\nWe tried to deliver your package today but no one was home. Print the attached shipping label to reschedule.", "Impersonates parcel delivery services."),
        ("https://voicemail-system.com/listen/msg-123", "Subject: New Voicemail from (Unknown Number) - 0:45s\n\nYou received a new voicemail. Click here to listen to the audio message.", "Impersonates PBX or VOIP system notifications."),
        ("https://e-ticket-portal.com/download/ticket", "Subject: Your Flight Itinerary and E-Ticket\n\nThank you for booking with us. Your e-ticket is attached. Please review your flight details.", "Targets travel arrangements."),
        ("https://secure-fax-service.com/view/fax", "Subject: Incoming Fax: 4 Pages from Medical Office\n\nYou have a new confidential fax. View the document securely online.", "Impersonates online fax services."),
        ("https://job-applicant-portal.com/resume/view", "Subject: Application for Marketing Manager position - Resume Attached\n\nHi, please find my resume attached for the open position. I look forward to hearing from you.", "Targets HR or hiring managers with fake resumes."),
        ("https://vendor-remittance.com/download/receipt", "Subject: Payment Remittance Advice\n\nPlease find attached the remittance advice for the payment processed today. Let us know if you have questions.", "Targets accounts receivable departments.")
    ],
    "Zero-day style tactics (new in 2025-2026)": [
        ("https://ai-assistant-auth.com/connect", "Subject: Authorize your Enterprise Copilot AI\n\nTo continue using your AI coding assistant, please re-authenticate your IDE integration via the new secure portal.", "Targets developers using modern AI assistants (Copilot, etc)."),
        ("https://deepfake-verification.com/verify-audio", "Subject: Urgent: Verify Voice Authorization\n\nOur system detected a potential deepfake voice authorization attempt on your account. Listen to the recording to confirm if it was you.", "Exploits fears of deepfakes and AI voice cloning."),
        ("https://web3-wallet-sync.com/connect", "Subject: Critical Smart Contract Upgrade Required\n\nA vulnerability was found in the decentralized exchange router. Upgrade your wallet connection to protect your assets.", "Targets crypto/Web3 users with technical jargon."),
        ("https://quantum-encryption-update.com/install", "Subject: Upgrade to Post-Quantum Cryptography\n\nAs part of our new security compliance, all employees must install the post-quantum encryption certificate. Download here.", "Uses emerging tech buzzwords (Quantum) to sound authoritative."),
        ("https://metaverse-office.com/meeting-room", "Subject: Invitation to VR Strategy Session\n\nYou've been invited to the virtual strategy session. Connect your VR headset to the enterprise metaverse portal.", "Exploits enterprise adoption of VR/Metaverse tools."),
        ("https://ai-generated-content-flag.com/appeal", "Subject: Your content has been flagged as AI-generated\n\nYour recent submission was flagged by our AI detectors. If this is a mistake, you must appeal within 24 hours to avoid a ban.", "Exploits policies around AI-generated content."),
        ("https://neural-link-diagnostic.com/run", "Subject: Routine BCI Diagnostic Check\n\nPlease run the diagnostic check on your Brain-Computer Interface device to ensure firmware is up to date.", "Highly futuristic, targeting early adopters of new hardware."),
        ("https://satellite-internet-config.com/update", "Subject: Starlink/LEO Satellite Firmware Update\n\nTo maintain optimal connection speeds on your satellite internet dish, please install this firmware update.", "Targets users of LEO satellite internet services."),
        ("https://autonomous-vehicle-portal.com/recall", "Subject: URGENT: Software Recall for your Autonomous Vehicle\n\nA critical bug in the self-driving software has been identified. Click to schedule an over-the-air update.", "Targets owners of autonomous vehicles."),
        ("https://nft-airdrop-claim.com/wallet", "Subject: You are eligible for the 2026 Protocol Airdrop\n\nClaim your tokens by verifying your wallet address. This is a limited time offer for early protocol adopters.", "Exploits ongoing crypto trends and airdrop farming.")
    ]
}

# Generate more samples to reach 75 if needed by recombining or just duplicating with slight changes, 
# but we have 7 categories * 10 samples = 70 samples. Let's add 5 more to reach 75.

extra_samples = [
    ("https://security-update.apple-icloud-login.com", "Subject: iCloud Storage Full\n\nYour iCloud storage is full. Upgrade now for 50GB free.", "Mixed legitimate + malicious content"),
    ("https://www.micros0ft-support.com", "Subject: Windows License Expired\n\nRenew your Windows license immediately.", "Subtle homoglyphs & obfuscated URLs"),
    ("https://hr-portal.com/mandatory-training", "Subject: Mandatory Compliance Training\n\nComplete your 2026 compliance training by Friday.", "Authority/Urgency manipulation"),
    ("https://goo.gl/quick-link", "Subject: checking in\n\nAre you available for a quick chat?", "Low-entropy / conversational phishing"),
    ("https://vendor-invoices.com/pay/10293", "Subject: Overdue Invoice 10293\n\nPlease pay the attached invoice to avoid late fees.", "Spear-phishing style (personalized)")
]

for cat, samps in categories.items():
    for url, body, technique in samps:
        data.append({
            "url": url,
            "email_body": body,
            "label": 1,
            "category": cat,
            "evasion_technique": technique
        })

# Add the extra 5
for url, body, cat in extra_samples:
    data.append({
        "url": url,
        "email_body": body,
        "label": 1,
        "category": cat,
        "evasion_technique": "Variation of standard techniques"
    })

df = pd.DataFrame(data)
df.to_csv("c:/Users/bhara/OneDrive/Tài liệu/PAD.ai/data/adversarial_samples.csv", index=False)
print("Saved 75 adversarial samples.")
