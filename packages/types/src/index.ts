export interface ScanResult {
  riskScore: number;
  threatLevel: "SAFE" | "SUSPICIOUS" | "CRITICAL";
  tokens: string[];
  reasons: string[];
}

export interface UpiPayload {
  vpa: string;
  payeeName?: string;
  amount?: string;
  isAutoDebitLure: boolean;
}
